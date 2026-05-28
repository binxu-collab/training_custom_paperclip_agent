import asyncio
import json
import logging
from asyncio import Lock
from datetime import datetime
from typing import Any

import aiofiles
import aiohttp
from gxl_inference_client.agent import Agent
from shared.core.environment import get_inference_url, get_timeout

logger = logging.getLogger(__name__)

# Global lock for JSONL file writes
_jsonl_write_lock = Lock()


async def _write_result_to_jsonl(result: dict[str, Any], jsonl_path: str) -> None:
    """Write a result dictionary to the JSONL file with thread-safe locking.

    Args:
        result: The result dictionary to write
        jsonl_path: Path to the JSONL file
    """
    async with _jsonl_write_lock:
        try:
            async with aiofiles.open(jsonl_path, mode="a") as f:
                await f.write(json.dumps(result) + "\n")
                await f.flush()
        except Exception as e:
            logger.error(f"Failed to write to JSONL file {jsonl_path}: {e}")


async def _process_single_entity_query(
    entity: str,
    query: str,
    agent_config: str,
    agent_id: str,
    session_id: str,
    jsonl_path: str | None = None,
    base_url: str | None = None,
    timeout: float = 5000.0,
    parent_agent_id: str | None = None,
) -> dict[str, Any]:
    """Process a single entity query using Agent class.

    Creates an agent for the given entity and queries it.

    Args:
        entity: Entity data to pass to the agent (e.g., URL, ID, path)
        query: The query to ask the agent
        agent_config: Agent configuration to use (e.g., "uniprot_id", "fda_query")
        agent_id: Unique ID to assign to the agent
        session_id: Session ID for shared file sandbox
        jsonl_path: Optional path to JSONL file to write results to
        base_url: Inference service URL
        timeout: HTTP request timeout in seconds (default: 5000)
        parent_agent_id: Optional ID of the parent agent

    Returns:
        Dictionary with entity, response, status, error, duration_seconds, etc.
    """
    start_time = datetime.now()
    result: dict[str, Any] = {
        "entity": entity,
        "response": None,
        "status": "pending",
        "start_time": start_time.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "error": None,
    }

    agent = None

    try:
        logger.info(f"[{agent_id}] Starting entity query")

        # Resolve base_url from environment if not provided
        resolved_base_url = base_url or get_inference_url()

        # Create Agent for this entity
        agent = await Agent.from_config(
            agent_config=agent_config,
            agent_id=agent_id,
            session_id=session_id,
            base_url=resolved_base_url,
            timeout=timeout,
            parent_agent_id=parent_agent_id,
        )

        # Prepend the entity context to the query
        modified_query = f"Entity: {entity}\n\nQuery: {query}"

        # Call agent with modified query
        logger.debug(f"[{agent_id}] Calling inference service...")
        response = await agent.call_async(modified_query)

        # Update result
        end_time = datetime.now()
        result.update(
            {
                "response": response,
                "status": "completed",
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
            }
        )

        logger.info(f"[{agent_id}] ✓ Completed in {result['duration_seconds']:.2f}s")

    except Exception as e:
        end_time = datetime.now()
        import traceback

        error_traceback = traceback.format_exc()
        logger.error(f"[{agent_id}] ✗ Error occurred:\n{error_traceback}")

        # Build detailed error message
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "agent_id": agent_id,
            "base_url": base_url or get_inference_url(),
            "timeout": timeout,
        }
        if agent:
            error_details.update(
                {
                    "agent_base_url": agent.base_url,
                    "agent_timeout": agent.timeout,
                    "agent_tools_loaded": (
                        len(agent.tool_list) if agent.tool_list else 0
                    ),
                }
            )

        result.update(
            {
                "status": "error",
                "error": str(e),
                "error_details": error_details,
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
            }
        )
        logger.error(f"[{agent_id}] ✗ Error details: {error_details}")

    finally:
        # Close agent properly in async context
        if agent is not None and agent._client is not None:
            try:
                await agent._client.close()
            except Exception as e:
                logger.warning(f"[{agent_id}] Failed to close agent client: {e}")

        # Write result to JSONL if path provided
        if jsonl_path:
            await _write_result_to_jsonl(result, jsonl_path)

    return result


async def query_agents_parallel(
    entities: list[str],
    agent_ids: list[str],
    query: str,
    agent_config: str,
    session_id: str = "default",
    jsonl_path: str | None = None,
    base_url: str | None = None,
    max_concurrent: int = 50,
    timeout: float = 5000.0,
    parent_agent_id: str | None = None,
) -> list[dict[str, Any]]:
    """Query multiple entities in parallel using agents.

    Each agent is spawned with a specific entity and queries it.

    Args:
        entities: List of entity data (e.g., URLs, IDs, paths)
        agent_ids: List of unique agent IDs to use for each agent. Must match length of entities.
        query: The query to ask each agent
        agent_config: Agent configuration to use (e.g., "uniprot_expert", "fda_query", "paper")
        session_id: Session ID for shared file sandbox
        jsonl_path: Optional path to JSONL file to write results to
        base_url: Inference service URL
        max_concurrent: Maximum number of concurrent query operations
        timeout: HTTP request timeout in seconds (default: 5000)
        parent_agent_id: Optional ID of the parent agent

    Returns:
        List of result dictionaries (one per entity)
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"🤖 Agent Swarm - Parallel Query")
    logger.info(f"{'=' * 60}")
    logger.info(f"Entities: {len(entities)}")
    logger.info(f"Query: {query[:100]}...")
    logger.info(f"Max concurrent: {max_concurrent}")
    logger.info(f"Agent config: {agent_config}")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Parent Agent ID: {parent_agent_id}")
    logger.info(f"{'=' * 60}\n")

    if len(entities) != len(agent_ids):
        raise ValueError(
            f"Length mismatch: entities ({len(entities)}) != agent_ids ({len(agent_ids)})"
        )

    # Process in batches if needed
    all_results = []

    for batch_start in range(0, len(entities), max_concurrent):
        batch_entities = entities[batch_start : batch_start + max_concurrent]
        batch_agent_ids = agent_ids[batch_start : batch_start + max_concurrent]

        logger.info(
            f"Processing batch {batch_start // max_concurrent + 1} ({len(batch_entities)} entities)..."
        )

        # Create tasks for this batch
        tasks = [
            _process_single_entity_query(
                entity=entity,
                query=query,
                agent_config=agent_config,
                agent_id=aid,
                session_id=session_id,
                jsonl_path=jsonl_path,
                base_url=base_url,
                timeout=timeout,
                parent_agent_id=parent_agent_id,
            )
            for entity, aid in zip(batch_entities, batch_agent_ids)
        ]

        # Run in parallel
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error dicts
        for i, r in enumerate(batch_results):
            if isinstance(r, BaseException):
                all_results.append(
                    {
                        "entity": batch_entities[i],
                        "response": None,
                        "status": "exception",
                        "error": str(r),
                        "start_time": None,
                        "end_time": None,
                        "duration_seconds": None,
                    }
                )
            elif isinstance(r, dict):
                all_results.append(r)
            else:
                all_results.append(
                    {
                        "entity": batch_entities[i],
                        "response": None,
                        "status": "error",
                        "error": f"Unexpected result type: {type(r)}",
                        "start_time": None,
                        "end_time": None,
                        "duration_seconds": None,
                    }
                )

    # Print summary
    completed = sum(1 for r in all_results if r.get("status") == "completed")
    errors = sum(1 for r in all_results if r.get("status") in ["error", "exception"])

    logger.info(f"\n{'=' * 60}")
    logger.info(f"✓ Agent Swarm completed!")
    logger.info(f"  Successful: {completed}/{len(all_results)}")
    logger.info(f"  Failed: {errors}/{len(all_results)}")
    logger.info(f"{'=' * 60}\n")

    return all_results


async def save_final_response_md(
    session_manager,
    session_id: str,
    subagent_id: str,
    response_text: str,
    api_key: str | None = None,
) -> str | None:
    """
    Save a subagent's final response as a markdown file in the session root directory.

    This function saves the final response to {subagent_id}_final_response.md in the
    session root (not in subagent folders) so that parent agents can cite it.

    Args:
        session_manager: Session manager instance
        session_id: Session ID for the sandbox
        subagent_id: ID of the subagent (used for filename)
        response_text: The final response text to save
        api_key: Optional API key for authentication

    Returns:
        Relative file path to the saved markdown file (e.g., "fda_docs_agent_abc123_final_response.md")
    """
    if not session_manager:
        logger.warning("No session manager available, cannot save final response")
        return None

    try:
        # Strip ---FINAL_RESPONSE--- marker if present
        cleaned_text = response_text
        marker = "---FINAL_RESPONSE---"
        if marker in cleaned_text:
            marker_index = cleaned_text.index(marker)
            cleaned_text = cleaned_text[marker_index + len(marker) :].lstrip()

        # Generate filename
        filename = f"{subagent_id}_final_response.md"

        # Save to session root (agent_id=None to save in root, not subagent folder)
        upload_result = await session_manager.upload_file_to_session_sandbox(
            session_id,
            filename,
            cleaned_text,
            api_key=api_key,
            agent_id=None,  # Save to root, not subagent folder
        )

        file_path = upload_result.get("file_path", filename)
        logger.info(f"Saved final response for {subagent_id} to {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Failed to save final response for {subagent_id}: {e}")
        return None


async def query_rest_api(
    endpoint: str,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    json_data: dict[str, Any] | None = None,
    description: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Async helper function to query REST APIs with consistent error handling.

    Args:
        endpoint: Full URL endpoint to query
        method: HTTP method ("GET" or "POST")
        params: Query parameters to include in the URL
        headers: HTTP headers for the request
        json_data: JSON data for POST requests
        description: Description of this query for error messages
        timeout: Request timeout in seconds (default: uses environment-aware config)

    Returns:
        Dictionary containing the result or error information:
        {
            "success": bool,
            "result": dict | None,
            "error": str | None,
            "query_info": dict | None
        }
    """
    # Set default headers if not provided
    if headers is None:
        headers = {"Accept": "application/json"}

    # Set default description if not provided
    if description is None:
        description = f"{method} request to {endpoint}"

    # Set default timeout from environment-aware config if not provided
    if timeout is None:
        timeout = get_timeout("GXL_HTTP_TIMEOUT", 15.0)

    try:
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            if method.upper() == "GET":
                async with session.get(
                    endpoint, params=params, headers=headers
                ) as response:
                    response.raise_for_status()
                    try:
                        result = await response.json()
                    except aiohttp.ContentTypeError:
                        # Return raw text if not JSON
                        result = {"raw_text": await response.text()}

                    return {
                        "success": True,
                        "query_info": {
                            "endpoint": endpoint,
                            "method": method,
                            "description": description,
                        },
                        "result": result,
                    }
            elif method.upper() == "POST":
                async with session.post(
                    endpoint, params=params, headers=headers, json=json_data
                ) as response:
                    response.raise_for_status()
                    try:
                        result = await response.json()
                    except aiohttp.ContentTypeError:
                        # Return raw text if not JSON
                        result = {"raw_text": await response.text()}

                    return {
                        "success": True,
                        "query_info": {
                            "endpoint": endpoint,
                            "method": method,
                            "description": description,
                        },
                        "result": result,
                    }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported HTTP method: {method}",
                    "query_info": {
                        "endpoint": endpoint,
                        "method": method,
                        "description": description,
                    },
                }

    except aiohttp.ClientError as e:
        error_msg = f"HTTP client error: {str(e)}"
        logger.error(f"{description} - {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "query_info": {
                "endpoint": endpoint,
                "method": method,
                "description": description,
            },
        }
    except TimeoutError:
        error_msg = f"Request timed out after {timeout}s"
        logger.error(f"{description} - {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "query_info": {
                "endpoint": endpoint,
                "method": method,
                "description": description,
            },
        }
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"{description} - {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "query_info": {
                "endpoint": endpoint,
                "method": method,
                "description": description,
            },
        }
