# CipherVerse - Frontend Architecture

This is the frontend component of CipherVerse, built with React, TypeScript, and Vite.

## Tech Stack

- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Tailwind CSS, `class-variance-authority`, `tailwind-merge`
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Data Fetching**: TanStack Query (React Query), Axios
- **Form Handling**: React Hook Form, Zod
- **Routing**: React Router DOM (v6)

## Architecture

The frontend uses a highly unified layout system for over 90 different cryptographic tools:

1. **`ToolPageLayout`**: The foundational wrapper enforcing the sleek 45/55 grid layout.
2. **`ToolInputPanel`**: Houses dynamic forms and options.
3. **`ToolResultPanel`**: Displays asynchronous outputs with built-in empty, loading, error, and success states.
4. **`useToolExecution`**: A custom wrapper around TanStack Query to seamlessly connect inputs to the FastAPI backend.

## Local Development

```bash
# Install dependencies
npm install

# Start the dev server
npm run dev
```

## Environment Variables

Make sure to specify the backend URL in your deployment platform:

```env
VITE_API_BASE_URL=http://localhost:8000  # Default local value
```
