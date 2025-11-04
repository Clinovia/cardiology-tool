import logging
import uuid

# configure logger (writes to project root: usage.log)
logging.basicConfig(
    filename="usage.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

def log_usage(tool_name, inputs, outputs=None):
    session_id = str(uuid.uuid4())[:8]  # short random session id
    logging.info(
        f"session={session_id} tool={tool_name} inputs={inputs} outputs={outputs}"
    )
