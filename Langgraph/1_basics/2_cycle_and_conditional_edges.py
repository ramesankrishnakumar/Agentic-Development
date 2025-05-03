from typing import TypedDict
from langgraph.graph import END, START, StateGraph


# state variable / type
class InputState(TypedDict):
    string_value: str
    numeric_value: int

# modifying the state
def modify_real_state(data: InputState):
    print(f"Current value: {data}")
    data["string_value"] = data["string_value"] + "!"
    data["numeric_value"] = data["numeric_value"] + 1
    return data

# router function
def router(data: InputState):
    if data["numeric_value"] < 5:
        return "branch_a"
    else:
        return "__end__"

# Directed Graph or State Machine
graph = StateGraph(InputState)

# creating nodes and edges
graph.add_node("branch_a", modify_real_state)
graph.add_node("branch_b", modify_real_state)
graph.add_edge(START, "branch_a")
graph.add_edge("branch_a", "branch_b")
graph.add_conditional_edges("branch_b", router, {"branch_a": "branch_a", "__end__": END})

graph.set_entry_point("branch_a")

runnable = graph.compile()

img_data = runnable.get_graph().draw_mermaid_png(output_file_path="conditional_graph.png")
# store the image in a file
with open("conditional_graph.png", "wb") as f:
    f.write(img_data)


value = runnable.invoke({"string_value": "hello", "numeric_value": 1})
print(value)
