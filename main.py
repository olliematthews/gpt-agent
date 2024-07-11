from openai import OpenAI


def main():
    client = OpenAI()
    print(
        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, what is your name"}],
        )
    )


if __name__ == "__main__":
    main()
