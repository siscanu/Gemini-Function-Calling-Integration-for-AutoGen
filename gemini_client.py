from autogen_core.models import ChatCompletionClient, LLMMessage, SystemMessage, UserMessage  #type: ignore
from typing import List, Optional
from google import genai # type: ignore
from autogen_core.tools import FunctionTool # type: ignore
from autogen_core import CancellationToken  # type: ignore

class GeminiChatCompletionClient(ChatCompletionClient):
    def __init__(self, model: str, api_key: str):
        # Initialize client with API key
        self._client = genai.Client(api_key=api_key)
        self.model_name = model
        
    async def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[FunctionTool]] = None,
        **kwargs
    ) -> LLMMessage:
        """Convert messages and get completion from Gemini"""
        # Convert messages to Gemini format
        prompt = self._convert_messages_to_prompt(messages)
        
        # Convert tools to Gemini's FunctionDeclaration format
        function_declarations = []
        if tools:
            for tool in tools:
                # Get schema from FunctionTool
                fn_schema = tool.schema
                
                # Create Gemini FunctionDeclaration
                function_declarations.append(
                    genai.types.FunctionDeclaration(
                        name=fn_schema['name'],
                        description=fn_schema['description'],
                        parameters=genai.types.Schema(
                            type=fn_schema['parameters']['type'],
                            properties={
                                k: genai.types.Schema(type=v['type'], description=v.get('description'))
                                for k, v in fn_schema['parameters']['properties'].items()
                            },
                            required=fn_schema['parameters'].get('required', [])
                        )
                    )
                )

        # Create tool config if tools are present
        config = None
        if tools:
            config = genai.types.GenerateContentConfig(
                tools=[genai.types.Tool(function_declarations=function_declarations)],
                tool_config=genai.types.ToolConfig(
                    function_calling_config=genai.types.FunctionCallingConfig(mode='AUTO')
                )
            )

        # Get response from Gemini using new SDK pattern
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        # Handle function calls if present
        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                # Execute function and get result
                tool = next((t for t in tools if t.schema['name'] == function_call.name), None)
                if tool:
                    # Create cancellation token for the tool execution
                    cancellation_token = CancellationToken()
                    result = await tool.run_json(function_call.args, cancellation_token)
                    # Send result back to model
                    messages.append(UserMessage(
                        content=str(result),
                        source="function"
                    ))
                    return await self.complete(messages, tools)
        
        # Return text response if no function call
        return UserMessage(
            content=response.text if response.text else "",
            source="assistant"
        )
    
    def _convert_messages_to_prompt(self, messages: List[LLMMessage]) -> str:
        """Convert AutoGen messages to Gemini-compatible format"""
        prompt = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt += f"System: {msg.content}\n\n"
            else:
                prompt += f"{msg.content}\n\n"
        return prompt 

    def actual_usage(self) -> dict:
        """Return dummy usage details."""
        return {}

    def capabilities(self) -> dict:
        """Return dummy model capabilities."""
        return {}

    def count_tokens(self, text: str) -> int:
        """Return a token count for given text."""
        return len(text.split())

    async def create(self, messages: List[LLMMessage], **kwargs) -> LLMMessage:
        """Delegate to complete() for creating a completion."""
        return await self.complete(messages, **kwargs)

    async def create_stream(self, messages: List[LLMMessage], **kwargs):
        """Yield a single completion as a dummy streaming implementation."""
        result = await self.complete(messages, **kwargs)
        yield result

    def model_info(self) -> str:
        """Return Gemini model information."""
        return f"Gemini model: {self.model_name}"

    def remaining_tokens(self) -> int:
        """Return a dummy remaining token count."""
        return 1024

    def total_usage(self) -> dict:
        """Return dummy total usage statistics."""
        return {"used": 0, "total": 0} 