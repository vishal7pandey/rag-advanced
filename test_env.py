import os
from dotenv import load_dotenv


def test_environment():
    # Load environment variables from .env file
    load_dotenv()

    # Check if required environment variables exist
    required_vars = ["OPENAI_API_KEY", "GEN_MODEL", "EMB_MODEL", "OFFLINE_MODE"]

    print("\n=== Environment Variables Check ===")
    for var in required_vars:
        value = os.getenv(var)
        status = "✅ Found" if value else "❌ Missing"
        print(
            f"{status} {var} = {value if var != 'OPENAI_API_KEY' else '***' + value[-4:] if value else None}"
        )

    # Test OpenAI connection if not in offline mode
    if os.getenv("OFFLINE_MODE", "").lower() != "true" and os.getenv("OPENAI_API_KEY"):
        print("\n=== Testing OpenAI Connection ===")
        try:
            import openai

            openai.api_key = os.getenv("OPENAI_API_KEY")
            models = openai.models.list()
            print("✅ Successfully connected to OpenAI API")
            print(f"Available models: {[m.id for m in models.data[:3]]}...")
        except Exception as e:
            print(f"❌ Error connecting to OpenAI: {str(e)}")
    else:
        print("\n=== Skipping OpenAI connection test (OFFLINE_MODE is True or no API key) ===")


if __name__ == "__main__":
    test_environment()
