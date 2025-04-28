## Hour 2: Agents and Tools in Langchain

### Concepts:

#### Agents:

Agents are systems that use an LLM to determine which actions to take. They have access to a set of tools and can execute them. The LLM acts as the agent's "brain," deciding the sequence of actions.

#### Agent Types:

Different agent types have different decision-making processes. A common one is the ReAct (Reasoning and Acting) agent, which interleaves reasoning (thinking step-by-step) and acting (using tools). Its iterative nature allows it to refine its approach based on tool outputs, making it particularly effective for complex tasks. Langchain supports various types, including Conversational Agents and agents leveraging model-specific function calling.

#### Tools:

Functions that an agent can use to interact with the external world. Tools can be anything from a simple calculator to a complex API call. Langchain provides many built-in tools, and you can create custom ones.

#### Custom Tools:

Defining your own functions and wrapping them so Langchain agents can understand and use them. This involves providing a name, description, and the function itself.

### Notes:

- The quality of the agent's decisions heavily depends on the LLM and the prompt used to guide it.
- Clear and descriptive tool descriptions are crucial for the LLM to understand when and how to use a tool.
- Tools should ideally be stateless and perform a specific, well-defined action.

### Exercise 2: Create a Custom Tool and Integrate with an Agent

1. Define a simple Python function that acts as a tool (e.g., a function that takes a city name and returns a hardcoded "weather" description).
2. Wrap this function as a Langchain Tool.
3. Initialize a Gemini LLM and an Agent that has access to this custom tool.
4. Give the agent a query that requires using the tool (e.g., "What is the weather in London?").
5. Run the agent and observe its steps.

### Quiz 2: Agents and Tools

- What is the main difference between a Langchain Chain and a Langchain Agent?
- What is the purpose of a Tool in the context of Langchain Agents?
- Describe the basic loop of an Agent's operation.
- Why is a good description important when defining a custom tool?

### Capstone Integration (Hour 2):

- **Identify the specific tools needed for the Research Assistant:**

  - A "Search" tool (initially simulated).

- **The LLM itself will be used for summarization and Q&A**, but we can think of these as actions the agent takes rather than separate "tools" in the Langchain sense for this simple setup, or we could wrap the summarization/Q&A logic as tools if preferred for modularity. For this plan, we'll focus on the "Search" tool as the primary external interaction.

- **Start defining the structure for the simulated "Search" tool.**
