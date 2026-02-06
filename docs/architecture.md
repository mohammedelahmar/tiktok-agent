# System Architecture

## Overview
TikTok Agent is built on a modern stack separating the heavy AI processing (Backend) from the interactive user experience (Frontend).

## Backend Architecture

The backend is a **FastAPI** application structured for modularity and scalability.

```
app/
├── main.py              # Application Entry Point & Configuration
├── schemas.py           # Pydantic Data Models (Request/Response)
├── routers/
│   └── jobs.py          # API Endpoints for Job Management
└── services/
    └── job_manager.py   # Business Logic & Persistence Layer
```

### Key Components

1.  **Job Manager (`app.services.job_manager`)**:
    -   Singleton service that manages the lifecycle of processing jobs.
    -   **Persistence**: Writes state to `jobs.json` to ensure durability across restarts.
    -   Handles updates from background workers.

2.  **Workers (Background Tasks)**:
    -   Processing is CPU-intensive, so it runs in a `ProcessPoolExecutor` (avoids blocking the async event loop).
    -   Workers (`web_api.py` wrappers) handle:
        -   Downloading (YouTube)
        -   AI Analysis (OpenCV/PyTorch)
        -   Rendering (MoviePy)

3.  **Data Flow**:
    -   Client POSTs to `/api/process/youtube` -> Job Created (Pending) -> Background Task Started.
    -   Client Polls `/api/status/{job_id}` -> Receives Pydantic Model with progress.
    -   Job Completes -> Results stored in `jobs.json` and returned to client.

## Frontend Architecture

The frontend is a **React** Single Page Application (SPA) built with **Vite**.

### Tech Stack
-   **Framework**: React 18
-   **Build Tool**: Vite
-   **Styling**: TailwindCSS + Custom Animations
-   **Icons**: Lucide React

### State Management
-   **Local State**: Uses `useState` and `useEffect` for managing the flow stages (`Input` -> `Processing` -> `Results` -> `Review`).
-   **Polling**: simpler than WebSockets for this use case. The client polls the status endpoint every 2 seconds during active processing.

### Component Structure
-   **`App.jsx`**: Main controller, handles routing between "pseudopages" (stages).
-   **`HistorySection.jsx`**: Fetches and displays the list of persistent jobs.
-   **`ResultsGrid.jsx`**: Displays viral candidates.
-   **`InputSection.jsx`**: Parameters form.

## Persistence Data Model (`jobs.json`)

The persistence file stores a flat dictionary of job objects:

```json
{
  "job-uuid-123": {
    "job_id": "job-uuid-123",
    "status": "completed",
    "created_at": 1700000000.0,
    "config": { ... },
    "result": {
      "success": true,
      "clips": [ ... ]
    }
  }
}
```
