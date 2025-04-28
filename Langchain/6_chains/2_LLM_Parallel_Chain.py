from dotenv import load_dotenv
import pprint
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")

string_parser = StrOutputParser()
json_parser = JsonOutputParser()

prompt_plan = ChatPromptTemplate.from_template(
    """
    Analyze the following REST API endpoint requirement and generate a concise plan
    outlining the responsibilities for the Controller, Service, and DAO layers in plain English.
    
    Framework: {framework}
    Endpoint Description: {endpoint_description}
    Data Object Name: {data_object_name}

    Output Instructions:
    - Provide a brief description for each layer's role in fulfilling the request.
    - Format the output as a JSON object with keys: "controller_contract", "service_contract", "dao_contract".
    - Example for "get item by id":
      {{
        "controller_contract": "Handle GET request at /items/{{item_id}}. Extract item_id from path. Validate item_id format (e.g., integer). Call service layer's get_item function with item_id. Return the found item or 404 error.",
        "service_contract": "Implement get_item(item_id) function. Call DAO layer's find_item_by_id function. Handle potential 'not found' scenarios from DAO and raise appropriate business exceptions or return None/empty.",
        "dao_contract": "Implement find_item_by_id(item_id) function. Query the data store (hint: {data_storage_hint}) for an item matching the given ID. Return the item data if found, otherwise return None."
      }}

    JSON Output:
    """
)

# Prompt 2: Generate Controller Code (using contract)
prompt_controller = ChatPromptTemplate.from_template(
    """
    Generate Python code for the Controller layer based *only* on the provided contract.

    Framework: {framework}
    Data Object Name: {data_object_name}
    Controller Contract: {contract}

    Instructions:
    - Implement the controller logic as described in the contract.
    - Include necessary imports for the specified framework.
    - Assume service layer functions exist as described in the contract.
    - Return *only* the Python code for the controller.

    Controller Code:
    """
)

# Prompt 3: Generate Service Code (using contract)
prompt_service = ChatPromptTemplate.from_template(
    """
    Generate Python code for the Service layer based *only* on the provided contract.

    Data Object Name: {data_object_name}
    Service Contract: {contract}

    Instructions:
    - Implement the service logic as described in the contract.
    - Assume DAO layer functions exist as described in the contract.
    - Define simple placeholder data models if needed based on Data Object Name.
    - Return *only* the Python code for the service.

    Service Code:
    """
)

# Prompt 4: Generate DAO Code (using contract)
prompt_dao = ChatPromptTemplate.from_template(
    """
    Generate Python code for the Data Access Object (DAO) layer based *only* on the provided contract.

    Data Object Name: {data_object_name}
    Data Storage Hint: {data_storage_hint}
    DAO Contract: {contract}

    Instructions:
    - Implement the DAO logic as described in the contract and storage hint.
    - Use placeholder logic for actual data store interaction if complex (e.g., "# TODO: DB query").
    - Return *only* the Python code for the DAO.

    DAO Code:
    """
)

# Prompt 5: Assemble the Code
prompt_assembly = ChatPromptTemplate.from_template(
    """
    You have been provided with generated code snippets for the Controller, Service, and DAO layers.
    Present these code snippets clearly, organized by layer.

    Framework: {framework}
    Original Endpoint Description: {endpoint_description}

    --- Controller Code ---
    ```python
    {controller_code}
    ```

    --- Service Code ---
    ```python
    {service_code}
    ```

    --- DAO Code ---
    ```python
    {dao_code}
    ```

    Add brief comments explaining how the layers might interact based on the generated code.
    """
)


# Helper function to prepare input for code generation prompts
def prepare_code_gen_input(input_dict, contract_key):
    return {
        "framework": input_dict["api_details"]["framework"],
        "data_object_name": input_dict["api_details"]["data_object_name"],
        "data_storage_hint": input_dict["api_details"]["data_storage_hint"],  # Needed for DAO
        "contract": input_dict["contract"][contract_key]  # Extract specific contract
    }


def prepare_assembly_input(input_dict):
    return {
        "framework": input_dict["api_details"]["framework"],
        "endpoint_description": input_dict["api_details"]["endpoint_description"],
        "controller_code": input_dict["code"]["controller_code"],
        "service_code": input_dict["code"]["service_code"],
        "dao_code": input_dict["code"]["dao_code"]
    }


api_details = {
    "framework": "Flask",
    "endpoint_description": "Retrieve a specific item using its unique ID.",
    "data_object_name": "Item",
    "data_storage_hint": "In-Memory Python Dictionary"
}


# --- Logging Function ---
def log_and_pass_through(stage_name):
    """Logs the data passing through and returns it unchanged."""
    def _log_func(data):
        print(f"\n--- Data entering stage: {stage_name} ---")
        # Use pprint for cleaner dictionary output
        pprint.pprint(data, indent=2)
        print(f"--- End of data for stage: {stage_name} ---\n")
        return data # Pass data through unchanged
    return RunnableLambda(_log_func)

# Chain 1: Planning - Generate the contracts
# Input: api_details -> Output: dictionary with contracts
planning_chain = prompt_plan | llm | json_parser

parallel_code_generator = RunnableParallel(
    controller_code=(
            RunnableLambda(
                lambda x: prepare_code_gen_input(x, "controller_contract"))  # "controller_contract" from prompt 1
            | prompt_controller
            | llm
            | string_parser
    ),
    service_code=(
            RunnableLambda(lambda x: prepare_code_gen_input(x, "service_contract"))
            | prompt_service
            | llm
            | string_parser
    ),
    dao_code=(
            RunnableLambda(lambda x: prepare_code_gen_input(x, "dao_contract"))
            | prompt_dao
            | llm
            | string_parser
    )
)

# { "framework", "endpoint_description", "data_object_name", "data_storage_hint" }

full_chain = (
        log_and_pass_through("Initial Input")  # Log initial input
        | RunnableParallel(
#{ "contract: {"controller_contract", "service_contract", "dao_contract"}","api_details: {framework, endpoint_description, data_object_name, data_storage_hint}"}
            contract=planning_chain,
            api_details=RunnablePassthrough()
        )
        | log_and_pass_through("After Planning & Passthrough")  # Log stage 1 output
#{ "code: {"controller_code", "service_code", "dao_code"}","contract: {"controller_contract", "service_contract", "dao_contract"}","api_details: {framework, endpoint_description, data_object_name, data_storage_hint}"}
        | RunnablePassthrough.assign(code=parallel_code_generator)
        | log_and_pass_through("After Planning & Passthrough")  # Log stage 1 output
        | RunnableLambda(lambda x: prepare_assembly_input(x))
        | prompt_assembly | llm | string_parser
)

final_output = full_chain.invoke(api_details)
print("\n--- Assembled Code Snippets ---")
print(final_output)
