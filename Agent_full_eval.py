
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import TopicAdherence, ToolCallF1
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from ragas.metrics.collections import ToolCallAccuracy

async def evaluate_topic_adherence():
    # Setup LLM
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    user_input = [
        HumanMessage(
            content="Can you provide me with details about Einstein's theory of relativity?"
        ),
        AIMessage(
            content="Sure, let me retrieve the relevant information for you.",
            tool_calls=[
                ToolCall(
                    name="document_search",
                    args={"query": "Einstein's theory of relativity"},
                )
            ],
        ),
        ToolMessage(
            content="Found relevant documents: 1. Relativity: The Special and the General Theory, 2. General Theory of Relativity by A. Einstein."
        ),
        AIMessage(
            content="I found some documents on Einstein's theory of relativity. Which one would you like to know more about: 'Relativity: The Special and the General Theory' or 'General Theory of Relativity by A. Einstein'?"
        ),
        HumanMessage(content="Tell me about the 'General Theory of Relativity'."),
        AIMessage(
            content="Got it! Let me fetch more details from 'General Theory of Relativity by A. Einstein'.",
            tool_calls=[
                ToolCall(
                    name="document_retrieve",
                    args={"document": "General Theory of Relativity by A. Einstein"},
                )
            ],
        ),
        ToolMessage(
            content="The document discusses how gravity affects the fabric of spacetime, describing the relationship between mass and spacetime curvature."
        ),
        AIMessage(
            content="The 'General Theory of Relativity' explains how gravity affects the fabric of spacetime and the relationship between mass and spacetime curvature. Would you like more details or a specific explanation?"
        ),
        HumanMessage(
            content="No, that's perfect. By the way, do you know any good recipes for a chocolate cake?"
        ),
        AIMessage(
            content="Sure! Let me find a simple and delicious recipe for a chocolate cake.",
            tool_calls=[
                ToolCall(name="recipe_search", args={"query": "chocolate cake recipe"})
            ],
        ),
        ToolMessage(
            content="Here's a popular recipe for a chocolate cake: Ingredients include flour, sugar, cocoa powder, eggs, milk, and butter. Instructions: Mix dry ingredients, add wet ingredients, and bake at 350°F for 30-35 minutes."
        ),
        AIMessage(
            content="I found a great recipe for chocolate cake! Would you like the full details, or is that summary enough?"
        ),
    ]

    # Evaluate with precision mode
    metric = TopicAdherence(llm=llm, mode="F1")
    result = await metric.ascore(
        user_input=user_input,
        reference_topics=["Einstein's theory of relativity", "chocolate cake recipe"],
        # reference_topics=["COOKING", "SCIENCE"],
    )
    return result


#-----------------------------------------
# Tools Call Accuracy



async def evaluate_tool_call_accuracy():
    # Define the conversation with tool calls
    user_input = [
        HumanMessage(content="What's the weather like in New York right now?"),
        AIMessage(
            content="The current temperature in New York is 75°F and it's partly cloudy.",
            tool_calls=[ToolCall(name="weather_check", args={"location": "New York"})],
        ),
        HumanMessage(content="Can you translate that to Celsius?"),
        AIMessage(
            content="Let me convert that to Celsius for you.",
            tool_calls=[
                ToolCall(
                    name="temperature_conversion", args={"temperature_fahrenheit": 75}
                )
            ],
        ),
        AIMessage(
            content="Sure, here's the UV index and air quality for New York.",
            tool_calls=[
                ToolCall(name="uv_index_lookup", args={"location": "New York"}),
            ],
        ),
    ]


    # Define expected tool calls
    reference_tool_calls = [
        ToolCall(name="weather_check", args={"location": "New York"}),
        ToolCall(name="temperature_conversion", args={"temperature_fahrenheit": 75}),
    ]

    # Evaluate

    # metric = ToolCallAccuracy(strict_order=True)    ---> it scores on exact match of tool calls
    metric = ToolCallAccuracy(strict_order=False)
    result = await metric.ascore(
        user_input=user_input,
        reference_tool_calls=reference_tool_calls,
    )
    print(f"Tool Call Accuracy: {result.value}")

    metric2 = ToolCallF1()
    result2 = await metric2.ascore(
        user_input=user_input,
        reference_tool_calls=reference_tool_calls,
    ) 
    print(f"Tool Call F1 Score: {result2.value}")
    return result, result2


# if __name__ == "__main__":
#     asyncio.run(evaluate_tool_call_accuracy())
