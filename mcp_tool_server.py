"""Local, read-only MCP tools. stdout is reserved for the SDK's stdio protocol."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import anyio
from jsonschema import Draft202012Validator, ValidationError
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool, ToolAnnotations

from knowledge_base import KNOWLEDGE_TOOL, KnowledgeBase, KnowledgeReadError
from local_embeddings import LocalEmbeddings
from orders import get_order


def create_server(knowledge_dir: Path, *, search_mode="keywords", embeddings_dir=None) -> Server:
    if search_mode not in ("keywords", "hybrid"):
        raise ValueError("search_mode must be keywords or hybrid")
    embeddings = (LocalEmbeddings(Path(embeddings_dir) if embeddings_dir is not None
                                  else Path(__file__).resolve().parent / ".local" / "embeddings")
                  if search_mode == "hybrid" else None)
    knowledge = KnowledgeBase(knowledge_dir, semantic_search=embeddings)
    order_schema = {"type": "object", "properties": {
        "order_id": {"type": "string", "minLength": 1, "maxLength": 100,
                     "description": "订单号，例如 O1003。"}},
        "required": ["order_id"], "additionalProperties": False}
    knowledge_schema = deepcopy(KNOWLEDGE_TOOL["function"]["parameters"])
    knowledge_schema["properties"]["query"]["minLength"] = 1
    definitions = [
        Tool(name="get_order", description="根据订单号查询模拟订单状态和物流信息，不能修改订单。",
             input_schema=order_schema, annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False)),
        Tool(name="search_knowledge", description=KNOWLEDGE_TOOL["function"]["description"],
             input_schema=knowledge_schema, annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False)),
    ]
    validators = {tool.name: Draft202012Validator(tool.input_schema) for tool in definitions}

    async def list_tools(context, params):
        return ListToolsResult(tools=definitions)

    async def call_tool(context, params):
        if params.name not in validators:
            return CallToolResult(is_error=True, content=[TextContent(type="text", text="未知或未授权的工具。")])
        arguments = params.arguments or {}
        try:
            validators[params.name].validate(arguments)
            key = "order_id" if params.name == "get_order" else "query"
            if not arguments[key].strip():
                raise ValueError("工具参数不能为空。")
            function = get_order if params.name == "get_order" else knowledge.search
            result = await anyio.to_thread.run_sync(function, arguments[key])
        except (ValidationError, TypeError, ValueError):
            return CallToolResult(is_error=True, content=[TextContent(type="text", text="工具参数不符合定义。")])
        except KnowledgeReadError:
            return CallToolResult(is_error=True, content=[TextContent(type="text", text="知识库资料无法读取。")])
        return CallToolResult(structured_content=result,
                              content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))])

    # The public low-level API lets us enforce the exact advertised JSON schemas,
    # including additionalProperties=False, without touching SDK internals.
    return Server("local-business-tools", version="1", on_list_tools=list_tools, on_call_tool=call_tool)


async def serve(knowledge_dir, *, search_mode="keywords", embeddings_dir=None):
    server = create_server(Path(knowledge_dir), search_mode=search_mode, embeddings_dir=embeddings_dir)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main():
    parser = argparse.ArgumentParser(description="Local read-only MCP tool service")
    parser.add_argument("--knowledge-dir", required=True, type=Path)
    parser.add_argument("--search-mode", choices=("keywords", "hybrid"), default="keywords")
    parser.add_argument("--embeddings-dir", type=Path)
    args = parser.parse_args()

    async def run():
        await serve(args.knowledge_dir, search_mode=args.search_mode, embeddings_dir=args.embeddings_dir)

    anyio.run(run)


if __name__ == "__main__":
    main()
