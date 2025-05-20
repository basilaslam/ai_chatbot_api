import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { RetrievalQAChain } from "langchain/chains";
import * as readline from 'readline';
import * as dotenv from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';

// Load environment variables
dotenv.config();

// Validate environment variables are present
const requiredEnvVars = [
  'OPENAI_API_KEY',
  'PINECONE_API_KEY',
  'PINECONE_INDEX',
  'PINECONE_ENV'
];

const missingEnvVars = requiredEnvVars.filter(varName => !process.env[varName]);
if (missingEnvVars.length > 0) {
  console.error(`Missing required environment variables: ${missingEnvVars.join(', ')}`);
  console.error('Please create a .env file with these variables.');
  process.exit(1);
}

// Create readline interface
function createInterface() {
  return readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });
}

// Function to ask a yes/no question
async function askYesNo(question: string): Promise<boolean> {
  const rl = createInterface();
  return new Promise((resolve) => {
    rl.question(`${question} (y/n): `, (answer) => {
      rl.close();
      resolve(answer.toLowerCase() === 'y' || answer.toLowerCase() === 'yes');
    });
  });
}

// Function to embed and store PDF
async function embedAndStorePDF(pdfPath: string): Promise<PineconeStore> {
  console.log(`Loading PDF from ${pdfPath}...`);
  const loader = new PDFLoader(pdfPath);
  const docs = await loader.load();
  console.log(`PDF loaded with ${docs.length} pages`);
  
  console.log("Splitting documents into chunks...");
  const splitter = new RecursiveCharacterTextSplitter({
    chunkOverlap: 20,
    chunkSize: 1000,
  });
  
  const splittedDocs = await splitter.splitDocuments(docs);
  console.log(`Created ${splittedDocs.length} text chunks`);
  
  console.log("Connecting to Pinecone...");
  const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY!,
  });

  console.log("Generating embeddings and storing in Pinecone...");
  const vectorStore = await PineconeStore.fromDocuments(
    splittedDocs,
    new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    }),
    {
      pineconeIndex: pinecone.Index(process.env.PINECONE_INDEX!),
      textKey: "text",
      namespace: "pdf-docs",
    }
  );

  console.log("Documents successfully embedded and stored in Pinecone");
  return vectorStore;
}

// Function to query existing vectors from Pinecone without re-embedding
async function getExistingVectorStore(): Promise<PineconeStore> {
  console.log("Connecting to Pinecone to retrieve existing embeddings...");
  const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY!,
  });

  // Initialize vector store from existing index
  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    }),
    {
      pineconeIndex: pinecone.Index(process.env.PINECONE_INDEX!),
      textKey: "text",
      namespace: "pdf-docs",
    }
  );
  
  console.log("Connected to existing vector store in Pinecone");
  return vectorStore;
}

// Function to query the vector store
async function queryVectorStore(vectorStore: PineconeStore, query: string) {
  console.log("Processing your query...");
  
  const retriever = vectorStore.asRetriever({ k: 4 });
  
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPENAI_API_KEY,
    temperature: 0,
  });

  const chain = RetrievalQAChain.fromLLM(model, retriever, {
    returnSourceDocuments: true,
  });

  const response = await chain.call({ query });
  return response;
}

// Print usage information
function printUsage() {
  console.log("\nUsage: npx ts-node pdf-chat.ts [options]");
  console.log("\nOptions:");
  console.log("  --help, -h                Display this help message");
  console.log("  --file, -f <path>         Specify the PDF file path");
  console.log("  --skip-embedding, -s      Skip embedding process and use existing vectors");
  console.log("\nExample:");
  console.log("  npx ts-node pdf-chat.ts --file ./documents/sample.pdf");
  console.log("  npx ts-node pdf-chat.ts -f ./documents/sample.pdf -s");
  process.exit(0);
}

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const options: { 
    filePath?: string; 
    skipEmbedding: boolean; 
  } = { 
    skipEmbedding: false 
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    
    if (arg === '--help' || arg === '-h') {
      printUsage();
    } else if (arg === '--file' || arg === '-f') {
      options.filePath = args[++i];
    } else if (arg === '--skip-embedding' || arg === '-s') {
      options.skipEmbedding = true;
    } else if (!options.filePath && !arg.startsWith('-')) {
      // Treat as file path if it's not an option and no file path set yet
      options.filePath = arg;
    }
  }

  return options;
}

// Main function
async function main() {
  try {
    // Parse command line arguments
    const options = parseArgs();
    let pdfPath = options.filePath;
    
    // If no file path provided, ask user for file path
    if (!pdfPath) {
      const rl = createInterface();
      pdfPath = await new Promise<string>((resolve) => {
        rl.question("Enter the path to your PDF file: ", (answer) => {
          rl.close();
          resolve(answer.trim());
        });
      });
    }

    // Validate the file exists
    if (!fs.existsSync(pdfPath!)) {
      console.error(`File not found: ${pdfPath}`);
      process.exit(1);
    }

    console.log(`\nSelected PDF file: ${pdfPath}`);
    
    // Initialize vector store
    let vectorStore: PineconeStore;
    
    if (options.skipEmbedding) {
      console.log("Skipping embedding process as requested...");
      vectorStore = await getExistingVectorStore();
    } else {
      // Ask if user wants to process/embed the PDF
      const shouldEmbed = await askYesNo("Do you want to process and embed this PDF into Pinecone?");
      
      if (shouldEmbed) {
        console.log("\nInitializing the PDF Chat system...");
        vectorStore = await embedAndStorePDF(pdfPath!);
      } else {
        console.log("\nSkipping embedding process...");
        console.log("Using existing vectors from Pinecone...");
        vectorStore = await getExistingVectorStore();
      }
    }
    
    console.log("\n===================================");
    console.log("PDF Chat System is ready!");
    console.log("Type your questions about the PDF below.");
    console.log("Type 'exit' or 'quit' to end the session.");
    console.log("===================================\n");
    
    const rl = createInterface();
    
    const askQuestion = () => {
      rl.question("Your question: ", async (query) => {
        // Check if user wants to exit
        if (query.toLowerCase() === 'exit' || query.toLowerCase() === 'quit') {
          console.log("Thank you for using PDF Chat. Goodbye!");
          rl.close();
          process.exit(0);
        }
        
        try {
          // Process the query
          const response = await queryVectorStore(vectorStore, query);
          
          console.log("\n----- ANSWER -----");
          console.log(response.text);
          console.log("------------------\n");
          
          // Ask for the next question
          askQuestion();
        } catch (error) {
          console.error("Error processing your query:", error);
          askQuestion();
        }
      });
    };
    
    // Start the question-answer loop
    askQuestion();
    
  } catch (error) {
    console.error("Error in main process:", error);
    process.exit(1);
  }
}

// Run the main function
main().catch(console.error);