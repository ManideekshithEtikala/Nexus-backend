-- Create conversation_summaries table for long-term memory storage
CREATE TABLE IF NOT EXISTS conversation_summaries (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    covered_message_count INTEGER NOT NULL,
    last_message_id INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_summaries_session_created 
  ON conversation_summaries(session_id, created_at DESC);
