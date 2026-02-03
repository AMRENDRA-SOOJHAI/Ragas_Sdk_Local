import os
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory

from ragas.metrics.collections import (
    ToolCallAccuracy,
    ToolCallF1,
    AgentGoalAccuracyWithReference,
    AgentGoalAccuracyWithoutReference,
)
from ragas.messages import HumanMessage, AIMessage, ToolCall, ToolMessage
api_key = os.environ.get("OPENAI_API_KEY")

def get_llm():
    client = AsyncOpenAI(api_key=api_key)
    return llm_factory("gpt-4o-mini", client=client)

#) Agent goal metrics need LLM
llm = get_llm()


async def evaluate_all_metrics(user_input, reference_tool_calls=None, goal_reference=None):
    results = {}

    #) ToolCallAccuracy
    metric_acc = ToolCallAccuracy(strict_order=False)
    acc_result = await metric_acc.ascore(
        user_input=user_input,
        reference_tool_calls=reference_tool_calls,
    )
    results["tool_call_accuracy"] = acc_result.value

    #) ToolCallF1
    metric_f1 = ToolCallF1()
    f1_result = await metric_f1.ascore(
        user_input=user_input,
        reference_tool_calls=reference_tool_calls,
    )
    results["tool_call_f1"] = f1_result.value

    # AgentGoalAccuracyWithReference
    metric_goal_ref = AgentGoalAccuracyWithReference(llm=llm)
    goal_ref_result = await metric_goal_ref.ascore(
        user_input=user_input,
        reference=goal_reference,
    )
    results["agent_goal_accuracy_with_reference"] = goal_ref_result.value


    # AgentGoalAccuracyWithoutReference
    metric_goal_no_ref = AgentGoalAccuracyWithoutReference(llm=llm)
    goal_no_ref_result = await metric_goal_no_ref.ascore(user_input=user_input)
    results["agent_goal_accuracy_without_reference"] = goal_no_ref_result.value

    return results


async def main():
    # user_input
    user_input = [
    HumanMessage(content="Book a table at best Chinese restaurant for 8:00pm"),

    AIMessage(
        content="Sure, I will search the best Chinese restaurants near you.",
        tool_calls=[
            ToolCall(
                name="restaurant_search",
                args={"cuisine": "Chinese", "time": "8:00pm"},
            )
        ],
    ),

    ToolMessage(content="Found: Golden Dragon, Jade Palace"),

    AIMessage(content="I found Golden Dragon and Jade Palace. Which one should I book?"),

    HumanMessage(content="Golden Dragon"),

    AIMessage(
        content="Okay, booking Golden Dragon for 8:00pm now.",
        tool_calls=[
            ToolCall(
                name="restaurant_book",
                args={"name": "Golden Dragon", "time": "8:00pm"},
            )
        ],
    ),

    ToolMessage(content="Table booked at Golden Dragon for 8:00pm."),

    AIMessage(content="Confirmed! Your table is booked at Golden Dragon for 8:00pm."),

    HumanMessage(content="thanks"),
]


    # expected tool calls for tool metrics
    reference_tool_calls = [
        ToolCall(name="restaurant_search", args={"cuisine": "Chinese", "time": "8:00pm"}),
        ToolCall(name="restaurant_book", args={"name": "Golden Dragon", "time": "8:00pm"}),
    ]

    #  reference goal (optional)
    goal_reference = "A table is booked at a Golden Dragon restaurant for 8:00pm"

    results = await evaluate_all_metrics(
        user_input=user_input,
        reference_tool_calls=reference_tool_calls,
        goal_reference=goal_reference,
    )

    print("\n All Tools Evaluation Metrics:")
    for k, v in results.items():
        print(f"{k}: {v}")

    return results

