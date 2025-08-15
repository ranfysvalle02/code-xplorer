## This Single Python File is a Self-Contained AI Code Analyst

We've all been there: starting a new job or joining a new team, you're faced with a sprawling, unfamiliar codebase. The `README` is a month out of date, and your only guide is the `git blame` history. Where do you even begin? The traditional approach involves hours of painstaking detective work, grepping for a `main` function and slowly piecing together the puzzle.

What if you could have an expert AI architect sit with you, point out the critical files, and then answer your questions with complete transparency? And what if this entire powerful experience was delivered in a **single, self-contained Python script** you can run locally against any codebase?

That’s not a future concept; it’s the reality of the `app.py` script below. It’s a complete, local-first web tool I built called **Code Explorer AI**, designed to transform AI from an opaque black box into a trustworthy, transparent partner for developers.

-----

### The Power of Being Local and Self-Contained

Before we dive into the AI magic, let's talk about the architecture, because it’s a core feature. The entire application—backend server, frontend interface, and all the logic—lives in one `app.py` file.

This isn't just a novelty; it's a design choice that prioritizes the developer experience:

  * **Zero-Friction Setup:** There's no complex installation, no Docker containers, no multi-service orchestration. With `#!/usr/bin/env python3` at the top, you can often make it executable and run it like any other command-line tool (`./app.py`). It’s the ultimate grab-and-go utility.
  * **Privacy and Security:** You can point it at a local directory containing proprietary code with confidence. The tool scans your files on your machine. Nothing is sent to an external service until you explicitly select code and ask a question.
  * **Ultimate Portability:** Drop this file into any project directory, run it, and you instantly have a dedicated analyst for that codebase. It’s a tool, not a cumbersome platform.

-----

### A New Workflow: The Three Steps to Code Clarity

Code Explorer AI guides you through a structured, three-step process that moves from a high-level overview to granular, specific questions, building trust at every stage.

#### Step 1: The AI Architect Gives You a Tour

When you first point the tool at a codebase (either a local path or a GitHub URL), it doesn't just present you with a wall of files. It uses an AI model, prompted as a "software architect," to perform its first crucial task: **intelligent file selection**.

It analyzes the entire file tree and identifies the 5-7 most important files for understanding the project's purpose—entry points, configurations, core business logic—and then explains *why* it chose them.

This is the AI's first act of transparency. Instead of you hunting for the starting point, the AI hands you a map and explains the key landmarks.

#### Step 2: The Developer Takes Command

The AI's recommendation is a starting point, not a mandate. The real power comes from the interactive context builder. The UI displays the full file tree, allowing you to expand files and see their high-level structure (classes and functions), which are parsed on the backend.

You are in complete control. You can select:

  * An entire file.
  * A specific class within a file.
  * A single function you need to understand.

This allows you to create a precise, surgical context for your questions. You're not just dumping thousands of lines of code into a prompt; you're focusing the AI's attention on exactly what matters to you.

#### Step 3: A Transparent Conversation with Reasoning

Once you've built your context, you can start the conversation. This is where the shift from "black box" to "glass box" becomes clear, thanks to Azure OpenAI's `responses` API.

When you ask a question like, "How does user authentication work based on the selected files?" the backend calls the model using `client.responses.create` with the `reasoning` parameter enabled.

```python
# A key snippet from the backend get_reasoned_llm_response function
response = client.responses.create(
    input=prompt_text,
    model=model_deployment,
    # This is the magic parameter!
    reasoning={"effort": "high", "summary": "detailed"}
)
```

The result is transformative. You don't just get a final answer. You get the answer *and* a step-by-step breakdown of the AI's thought process.

**Black Box AI:**

> "User authentication is handled by verifying a JWT token."

**Glass Box AI (with Reasoning):**

> **Reasoning:**
>
> 1.  *User asks about authentication.*
> 2.  *Scanning context, I see `auth_controller.py` and `jwt_service.py` were provided.*
> 3.  *In `auth_controller.py`, the `/login` route uses the `jwt_service.create_token()` method.*
> 4.  *In `jwt_service.py`, the `verify_token()` function checks the token's signature and expiration.*
>
> **Answer:**
> Based on the provided files, user authentication is managed through a JWT (JSON Web Token) system. The process begins in `auth_controller.py`...

This audit trail allows you to verify the AI's logic against the source code, building confidence and ensuring you can trust the conclusion.

----


## From Black Box to Glass Box: Building Trustworthy AI with Reasoning APIs

Artificial intelligence is phenomenal at analyzing complex information, but it has a trust problem. When you ask an AI to analyze a codebase, how do you know it's right? How did it arrive at its conclusion? If you can't see the "thought process," you're trusting a black box, which can be risky when dealing with intricate software projects.

This is where a new paradigm in AI interaction comes in: **reasoned responses**. Instead of just getting a final answer, you get the answer *and* a step-by-step breakdown of how the AI produced it. This creates an audit trail, transforming the AI from an opaque oracle into a transparent partner.

I explored this concept by building **Code Explorer AI**, a Flask and Python web tool that uses Azure OpenAI's new `responses` API to create a more transparent and controllable code analysis experience.

-----

### The Problem: Opaque AI is a Risky Co-Pilot

Imagine asking an AI, "Refactor this function to be more efficient." It gives you a new block of code. Do you trust it blindly and commit it to your main branch? Probably not. You'd want to know:

  * What logic did it identify as inefficient?
  * What assumptions did it make about the data?
  * Did it consider any edge cases?

Without this reasoning, the AI's suggestion is just a "trust me" proposition. A visible reasoning process, however, gives you the power to audit its logic, verify its claims against the source code, and ultimately make an informed decision. It's the key to building confidence and control. 🧠

-----

### Introducing the Azure OpenAI `responses` API

The magic behind this transparency is a different way of calling the model. Instead of the standard chat completions endpoint, we can use an endpoint designed specifically for reasoned, structured output. The new **Responses API** simplifies workflows that involve tool use, code execution and state management.

In the Code Explorer AI script, the core of this interaction happens in one function. It uses the `client.responses.create` method, which is distinct from the more common `client.chat.completions.create`.

```python
def get_reasoned_llm_response(client, prompt_text, model_deployment):
    """
    Calls a reasoning-focused endpoint, expecting a structured response
    with summaries and a final answer.
    """
    if not client:
        return {"answer": "[Error: OpenAI client not configured]", "summaries": []}
    try:
        # The key API call for reasoned responses
        response = client.responses.create(
            input=prompt_text,
            model=model_deployment,
            reasoning={"effort": "high", "summary": "detailed"} # Requesting the audit trail
        )

        response_data = response.model_dump()
        result = {"answer": "Could not extract answer.", "summaries": []}

        # The response payload is structured, not just a single string
        output_blocks = response_data.get("output", [])
        if output_blocks:
            summary_section = output_blocks[0].get("summary", [])
            if summary_section:
                # Extracting the step-by-step reasoning
                result["summaries"] = [s.get("text") for s in summary_section if s.get("text")]

            # The final answer is in a separate content block
            content_section_index = 1 if summary_section else 0
            if len(output_blocks) > content_section_index:
                result["answer"] = output_blocks[content_section_index]["content"][0].get("text", result["answer"])

        return result
    except Exception as e:
        logging.error(f"Error in get_reasoned_llm_response: {e}")
        return {"answer": f"[Error calling LLM: {e}]", "summaries": []}
```

The crucial part is the `reasoning` parameter. By setting `summary: "detailed"`, we are explicitly asking the API to return a list of its internal reasoning steps alongside the final answer. The response is a structured JSON object, allowing us to cleanly separate the audit trail (`summaries`) from the conclusion (`answer`).

You can also control the amount of "effort" the model puts into its reasoning process. The `reasoning_effort` parameter can be set to **"low"**, **"medium"**, or **"high"**, which adjusts the computational depth the model uses to analyze the prompt. For critical tasks where accuracy is paramount, setting the effort to "high" is recommended.

-----

### Responses API vs. Chat Completions API: When to Choose Which

The **Chat Completions API** is the industry standard for building AI applications. It's a stateless API, meaning you need to send the entire conversation history with each request. This is great for simple, single-turn interactions, but can become cumbersome for more complex, multi-turn conversations, especially when using tools.

The new **Responses API** is a stateful API that manages the conversation history for you. This makes it much easier to build complex, multi-turn applications that use tools and require a high degree of reasoning. Here are some key differences:

| Feature | Chat Completions API | Responses API |
| --- | --- | --- |
| **State Management** | Stateless | Stateful |
| **Conversation History** | Managed by the developer | Managed by OpenAI |
| **Best For** | Simple, single-turn interactions | Complex, multi-turn interactions with tools and reasoning |
| **Tools** | Supported, but requires more developer effort | Simplified tool use and execution |

In short, if you're building a simple chatbot, the Chat Completions API is a great choice. But if you're building a more complex application that requires reasoning, tool use, and state management, the Responses API is the way to go. 🚀
