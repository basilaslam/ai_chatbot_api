# PDF Chat CLI

A terminal-based application for querying and chatting with PDF documents using natural language. Built with LangChain, OpenAI, and Pinecone vector database.

## Overview

PDF Chat CLI allows you to:

1. Load and process PDF documents
2. Convert document content into vector embeddings using OpenAI's embedding model
3. Store these embeddings in Pinecone vector database
4. Query the document content using natural language questions
5. Get AI-generated answers based on the document content

The system uses Retrieval Augmented Generation (RAG) to provide accurate answers directly from your PDF documents.

## Requirements

- Node.js (v14+)
- TypeScript
- OpenAI API key
- Pinecone account and API key

## Installation

1. Clone the repository or download the source code

2. Install dependencies:
   ```bash
   npm install @langchain/community langchain @langchain/openai @pinecone-database/pinecone readline dotenv
   ```

3. Create a `.env` file in the project root with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX=your_pinecone_index_name
   PINECONE_ENV=your_pinecone_environment
   ```

## Usage

### Basic Usage

```bash
npx ts-node pdf-chat.ts --file ./path/to/your/document.pdf
```

### Command Line Options

```
Options:
  --help, -h                Display help message
  --file, -f <path>         Specify the PDF file path
  --skip-embedding, -s      Skip embedding process and use existing vectors

Example:
  npx ts-node pdf-chat.ts --file ./documents/sample.pdf
  npx ts-node pdf-chat.ts -f ./documents/sample.pdf -s
```

### Interactive Mode

If you run the script without specifying a file path, it will prompt you to enter one:

```bash
npx ts-node pdf-chat.ts
```

### Embedding Options

- On first use with a PDF, the system will ask if you want to process and embed the document
- For subsequent queries, you can use the `--skip-embedding` flag to reuse existing embeddings

## How It Works

1. **Document Processing**: The PDF is loaded and split into manageable text chunks
2. **Embedding Generation**: Each text chunk is converted into a vector embedding using OpenAI's embedding model
3. **Vector Storage**: The embeddings are stored in Pinecone with the original text as metadata
4. **Query Processing**: When you ask a question, it's converted to a vector embedding
5. **Semantic Search**: The system finds the most relevant document chunks by comparing vector similarity
6. **Answer Generation**: OpenAI's GPT model generates an answer based on the retrieved context

## Example Session

```
Selected PDF file: ./documents/annual_report.pdf
Do you want to process and embed this PDF into Pinecone? (y/n): y

Initializing the PDF Chat system...
Loading PDF from ./documents/annual_report.pdf...
PDF loaded with 25 pages
Splitting documents into chunks...
Created 47 text chunks
Connecting to Pinecone...
Generating embeddings and storing in Pinecone...
Documents successfully embedded and stored in Pinecone

===================================
PDF Chat System is ready!
Type your questions about the PDF below.
Type 'exit' or 'quit' to end the session.
===================================

Your question: What were the company's revenue figures for last year?

----- ANSWER -----
According to the annual report, the company's revenue for last year was $24.5 million, which represents a 12% increase from the previous year's revenue of $21.9 million.
------------------

Your question: exit
Thank you for using PDF Chat. Goodbye!
```

## Troubleshooting

- **Missing Environment Variables**: Ensure all required environment variables are set in your `.env` file
- **File Not Found**: Verify the path to your PDF file is correct
- **Connection Issues**: Check your internet connection and Pinecone service status
- **API Limits**: Be aware of rate limits on both OpenAI and Pinecone services

## License

MIT