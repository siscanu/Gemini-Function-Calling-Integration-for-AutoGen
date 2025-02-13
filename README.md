# Gemini Function Calling Integration for AutoGen

This implementation provides a custom integration of Google's Gemini model with AutoGen's function calling system. It allows Gemini to seamlessly work with AutoGen's tools and agent framework while maintaining Gemini's native function calling capabilities.

## Overview

The `GeminiChatCompletionClient` class implements AutoGen's `ChatCompletionClient` interface, enabling Gemini to:
- Work with AutoGen's tool system
- Handle function calling in a way compatible with both frameworks
- Maintain conversation context
- Execute tools and process their results

## Key Features

- **AutoGen Compatibility**: Implements all required methods from `ChatCompletionClient`
- **Native Gemini Integration**: Uses Gemini's function calling system directly
- **Tool Conversion**: Automatically converts AutoGen tools to Gemini function declarations
- **Recursive Function Handling**: Supports chained function calls and result processing
- **Error Handling**: Robust error checking and response validation

## Usage

```python
from autogen_core.models import ChatCompletionClient, LLMMessage
from google import genai
from autogen_core.tools import FunctionTool

# Initialize the client
client = GeminiChatCompletionClient(
    model="gemini-pro",  # or your chosen Gemini model
    api_key="your-api-key"
)

# Create a tool
async def get_time(timezone: str = "UTC") -> dict:
    """Get current time in specified timezone."""
    # Tool implementation
    pass

time_tool = FunctionTool(
    get_time,
    name="get_time",
    description="Get current time in any timezone"
)

# Use in conversation
response = await client.complete(
    messages=[...],  # Your conversation messages
    tools=[time_tool]  # Your tools
)
```

## Implementation Details

### Tool Conversion
The client converts AutoGen's tools to Gemini's function declaration format:

```python
# AutoGen tool schema
{
    'name': 'get_time',
    'description': 'Get current time in any timezone',
    'parameters': {
        'type': 'object',
        'properties': {
            'timezone': {
                'type': 'string',
                'description': 'Timezone name'
            }
        }
    }
}

# Converted to Gemini format
genai.types.FunctionDeclaration(
    name='get_time',
    description='Get current time in any timezone',
    parameters=genai.types.Schema(...)
)
```

### Function Call Handling

1. **Detection**: Checks for function calls in Gemini's response
2. **Execution**: Runs the corresponding AutoGen tool
3. **Result Processing**: Feeds results back to Gemini
4. **Recursion**: Handles chained function calls automatically

### Message Conversion

Converts AutoGen's message format to Gemini's expected format:
- System messages prefixed with "System: "
- User and assistant messages preserved as-is
- Maintains conversation context

## Key Methods

- `complete()`: Main method for generating responses and handling function calls
- `_convert_messages_to_prompt()`: Converts AutoGen messages to Gemini format
- `create()`: Wrapper for complete() to match AutoGen's interface
- `create_stream()`: Basic streaming implementation

## Integration with AutoGen Agents

This client can be used with any AutoGen agent:

```python
class MyAgent(RoutedAgent):
    def __init__(self, name: str):
        self._model_client = GeminiChatCompletionClient(...)
        self._tools = [your_tools]
        
    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> Message:
        response = await self._model_client.complete(
            messages=[...],
            tools=self._tools
        )
        return Message(content=response.content)
```

## Code Walkthrough

### 1. Tool Schema Conversion
```python
# Convert AutoGen tools to Gemini's FunctionDeclaration format
function_declarations = []
if tools:
    for tool in tools:
        fn_schema = tool.schema
        function_declarations.append(
            genai.types.FunctionDeclaration(
                name=fn_schema['name'],
                description=fn_schema['description'],
                parameters=genai.types.Schema(...)
            )
        )
```

### 2. Function Call Configuration
```python
config = genai.types.GenerateContentConfig(
    tools=[genai.types.Tool(function_declarations=function_declarations)],
    tool_config=genai.types.ToolConfig(
        function_calling_config=genai.types.FunctionCallingConfig(mode='AUTO')
    )
)
```

### 3. Function Call Processing
```python
if response.candidates and response.candidates[0].content.parts:
    part = response.candidates[0].content.parts[0]
    if hasattr(part, 'function_call') and part.function_call:
        function_call = part.function_call
        tool = next((t for t in tools if t.schema['name'] == function_call.name), None)
        if tool:
            result = await tool.run_json(function_call.args, cancellation_token)
            messages.append(UserMessage(content=str(result), source="function"))
            return await self.complete(messages, tools)
```

## Limitations

- Streaming is implemented as a basic wrapper
- Usage metrics are placeholder implementations
- Token counting is simplified
- Some Gemini-specific features might not be exposed through the AutoGen interface

## Requirements

- AutoGen Core
- Google Generative AI Python SDK
- Python 3.8+
- Async IO support

## Contributing

Feel free to contribute improvements or report issues. Some areas that could be enhanced:
- Better streaming support
- More accurate token counting
- Full usage metrics
- Additional Gemini-specific features

## License

MIT License
