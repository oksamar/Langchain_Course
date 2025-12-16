# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def main():
    print("Hello from langchain-course!!!")

    information = """
    Elon Reeve Musk (born June 28, 1971) is a businessman and entrepreneur known
    for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest
    person in the world since 2021; as of December 2025, Forbes estimates his
    net worth to be around $677 billion.
    """

    summary_template = """
    Given the following information about a person:

    {information}

    Please provide:
    1. A short summary
    2. Two interesting facts
    """

    prompt = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    '''llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini"   # safer choice
    )'''

    llm = ChatOllama(
        temperature=0,
        model="gemma3:270m"   # safer choice
    )

    chain = prompt | llm

    response = chain.invoke({"information": information})

    print("\n--- Model Response ---")
    print(response.content)


if __name__ == "__main__":
    main()
