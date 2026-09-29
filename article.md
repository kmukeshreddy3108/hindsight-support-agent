# How Hindsight Fixed My Agent's Repeat Incident Blindspot

Stateless LLMs make terrible support engineers because they treat every incoming ticket like day zero. If a customer files three consecutive tickets about transient database timeouts over two days, a standard automated router will evaluate the third ticket in complete isolation, categorize it as a low-severity query, and leave the customer waiting in an unassigned queue.

I spent months fighting this exact failure mode in automated ticket triage. In this post, I will walk through how we redesigned our tier-1 support routing pipeline around persistent episodic memory using [Hindsight](https://github.com/vectorize-io/hindsight), how the retain-and-recall pattern replaced clunky context stuffing, and what we learned when running stateful agent memory in production.

---

## The Stateless Triage Problem

When an automated support agent processes an incoming incident, its job is straightforward: parse the customer's text, extract the root component, estimate business impact, and assign a priority level (P1 through P4) along with the appropriate routing tag.

In isolation, modern language models do this reasonably well. If a ticket says "Our production cluster is completely unresponsive," the model correctly flags it as P1. But enterprise support rarely presents itself so cleanly. More often, critical incidents start as subtle, recurring whispers:

1. **Monday 10:00 AM (`TCK-1001`)**: *"Seeing slight query latency on our secondary node under load."* $\rightarrow$ Classified as **P4 (Low)**.
2. **Tuesday 02:00 PM (`TCK-1004`)**: *"Connection pool timeout reset itself after five minutes."* $\rightarrow$ Classified as **P3 (Medium)**.
3. **Wednesday 09:30 AM (`TCK-1008`)**: *"Queries dropped again this morning during our batch sync."* $\rightarrow$ Classified as **P3 (Medium)**.

To a human engineer who worked yesterday's shift, ticket three is clearly not an isolated P3—it is an escalating infrastructure defect or a degraded storage volume that requires immediate escalation before the customer churns.

To a stateless LLM receiving only the payload of `TCK-1008`, however, there is zero historical signal. The ticket text looks mild, the immediate blast radius seems contained, and it gets routed to a standard backlog queue.

We initially attempted to fix this by querying our PostgreSQL ticket database and dumping the last five closed tickets directly into the prompt. That strategy degraded quickly:
- **Context bloat**: Dumping raw transcripts pushed prompt tokens through the roof and introduced severe latency on every webhook invocation.
- **Noise over signal**: The LLM frequently latched onto irrelevant details from resolved billing or onboarding tickets included in the historical dump.
- **Lack of semantic continuity**: Keyword matching against past tickets missed related problems described in different words (e.g., "Postgres connection dropped" vs. "504 Gateway Timeout on checkout").

We needed a dedicated memory layer that could retain structured incident episodes per customer and recall relevant context dynamically when a new ticket arrived.

---

## Architecture: Integrating Agent Memory into the Triage Pipeline

To solve this, we decoupled the agent's inference engine from its memory storage using [Vectorize agent memory](https://vectorize.io/what-is-agent-memory). Instead of brute-forcing past ticket transcripts into the prompt or relying on generic vector embeddings that lack temporal awareness, we structured our pipeline around Hindsight's retain-and-recall primitives.

```
                  ┌────────────────────────────────────────┐
                  │          Incoming Support Ticket       │
                  │  (User ID, Ticket ID, Issue Details)   │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │         Triage Worker Service          │
                  └─────────┬────────────────────┬─────────┘
                            │                    │
                1. Recall Past History   3. Retain New Event
                            │                    │
                            ▼                    ▼
                ┌──────────────────────────────────────────┐
                │        Hindsight Memory Engine           │
                │        Bank: `support-ticket-bank`       │
                │  - Hybrid Semantic & Temporal Index      │
                │  - Scoped per Tenant & Customer Entity   │
                └──────────────────┬───────────────────────┘
                                   │
                         2. Injected Context
                                   │
                                   ▼
                  ┌────────────────────────────────────────┐
                  │       Agent Decision Engine (LLM)      │
                  │  - Correlates Recalled Context         │
                  │  - Escalates Priority on Flapping Logs │
                  │  - Emits Route & Priority Tag          │
                  └────────────────────────────────────────┘
```

The system operates in three distinct phases for every incoming webhook:

1. **Episodic Recall**: Before evaluating severity, the worker queries Hindsight with a targeted hypothesis (e.g., *"Has this user experienced database performance issues or connection timeouts recently?"*).
2. **Contextual Evaluation**: The recalled context—summarized and distilled from past interactions—is injected into the system prompt alongside the current ticket body.
3. **Episodic Retention**: Once processed, the current ticket's core problem statement, environment metadata, and resolution state are written back to the memory bank for future triage cycles.

---

## Implementation Details

The implementation centers around a clean Python service interface interacting with the Hindsight bank API. Below are the core functions handling retention, retrieval, and decision orchestration.

### 1. Retaining Incident Episodes

Whenever a ticket is created or updated, we retain an episodic record within our customer support memory bank. Rather than dumping raw HTML emails or chat transcripts, we store normalized issue representations scoped to the customer entity.

```python
import os
import requests

HINDSIGHT_URL = os.getenv("HINDSIGHT_URL", "https://api.hindsight.vectorize.io")
BANK_ID = "support-ticket-bank"

def retain_ticket_memory(user_id: str, ticket_id: str, issue_details: str) -> None:
    """Persist structured customer support interactions into Hindsight memory."""
    url = f"{HINDSIGHT_URL}/banks/{BANK_ID}/retain"
    payload = {
        "content": f"User {user_id} reported issue on Ticket {ticket_id}: {issue_details}",
        "context": "customer_support",
        "metadata": {
            "user_id": user_id,
            "ticket_id": ticket_id
        }
    }
    
    response = requests.post(url, json=payload, timeout=5)
    response.raise_for_status()
```

By tagging memories with explicit context and customer identifiers, Hindsight indexes the event across both semantic and temporal dimensions without polluting unrelated global workspaces.

### 2. Recalling Relevant Incident Context

When a new ticket lands, we do not perform a dumb cosine similarity search over a flat vector table. Instead, we query Hindsight to extract whether the customer has a history of related architectural or infrastructure failures.

```python
def recall_ticket_memory(user_id: str, query: str) -> str:
    """Retrieve historical support context using Hindsight hybrid recall."""
    url = f"{HINDSIGHT_URL}/banks/{BANK_ID}/recall"
    payload = {
        "query": query,
        "filters": {"user_id": user_id},
        "top_k": 3
    }
    
    response = requests.post(url, json=payload, timeout=5)
    response.raise_for_status()
    
    data = response.json()
    return data.get("summary", "No prior related incidents found.")
```

Following the specifications in the [Hindsight documentation](https://hindsight.vectorize.io/), the recall endpoint synthesizes relevant past facts into an information-dense summary rather than returning fragmented text chunks that require secondary parsing.

### 3. Executing the Stateful Decision Workflow

With retain and recall in place, the core triage loop transforms from an isolated text classifier into a state-aware decision workflow:

```python
def process_agent_workflow(user_id: str, incoming_ticket_id: str, incoming_text: str):
    # Step 1: Query memory for prior related system instability
    recalled_context = recall_ticket_memory(
        user_id=user_id,
        query=f"Has user {user_id} experienced issues related to: {incoming_text}?"
    )
    
    # Step 2: Combine incoming ticket with recalled context for triage
    decision_prompt = f"""
    Current Ticket ID: {incoming_ticket_id}
    User ID: {user_id}
    Current Issue: {incoming_text}
    
    Historical Context:
    {recalled_context}
    
    Task: Evaluate ticket priority (P1, P2, P3, P4). If this issue represents
    a recurring or worsening failure mode based on historical context,
    escalate priority accordingly.
    """
    
    # (Inference call to LLM engine omitted for brevity)
    triage_result = {
        "priority": "HIGH (P1)",
        "reason": "Recurring database connection timeouts detected across multiple tickets."
    }
    
    # Step 3: Retain current incident into memory for subsequent turns
    retain_ticket_memory(user_id, incoming_ticket_id, incoming_text)
    
    return triage_result
```

---

## Live Incident Tracing: Catching a Flapping Service

To verify the practical impact of persistent memory, consider what happens when a real-world multi-day incident unfolds for customer `usr_402`.

### Episode 1: The Initial Flake
- **Ticket `TCK-1001`**: *"Database connection timeout under high traffic."*
- **Memory State**: Blank.
- **Recalled Context**: `"No prior related incidents found."`
- **Agent Action**: Categorized as **P3**. Standard troubleshooting steps sent. The event is stored in Hindsight: `User usr_402 reported issue on Ticket TCK-1001: Database connection timeout under high traffic.`

### Episode 2: The Second Occurrence
- **Ticket `TCK-1002`**: *"API server failed to respond during peak hours; check logs."*
- **Recalled Context via Hindsight**: `"Found past record: User usr_402 previously reported database connection timeouts under high traffic on Ticket TCK-1001."`
- **Agent Action**: The decision engine notes that the API failure coincides with the earlier database timeout pattern. Instead of assigning another P3, it assigns **P1 (High)** and directly routes the ticket to the infrastructure on-call engineer with the prior context pre-attached.

Instead of waiting for an angry escalation email after three failed triage attempts, the system intercepted the pattern on ticket two.

---

## Realities of Running Agent Memory in Production

Moving from stateless LLM calls to a stateful memory architecture introduced several non-obvious engineering challenges that are worth highlighting:

### 1. The Trap of Raw Transcript Ingestion
Early in development, we tried pushing raw ticket conversations directly into memory. This was a mistake. Raw chat contains pleasantries, signatures, automated bot replies, and boilerplate headers. Storing raw noise pollutes recall embeddings and degrades semantic match precision. We enforce a normalization step that strips metadata and extracts only the factual assertion before calling `retain()`.

### 2. Temporal Decay vs. Semantic Relevance
Not all past tickets remain relevant forever. A database issue from twelve months ago on an older major version should not necessarily escalate a minor query error today. Hindsight handles this by weighting recency alongside semantic relevance, ensuring that active incidents take precedence over ancient resolutions without requiring manual timestamp pruning scripts.

### 3. Isolation by Tenant and Entity
Memory must be strictly partitioned. We scope memory banks and filter lookups by explicit tenant and user IDs. Shared memory across different customer boundaries is an immediate compliance and security violation, while global memory banks within a single customer should only contain company-wide infrastructure changes rather than individual user complaints.

---

## Takeaways for Engineering Teams Building Agents

If you are transitioning support automation or internal developer agents from stateless prompts to persistent workflows, here are the core principles that made our system reliable:

1. **Prompt stuffing is an anti-pattern for state**: Shoving large conversation histories into every context window wastes money, increases TTFT (time-to-first-token), and causes attention dilution in LLMs.
2. **Episodic retention must be concise**: Store distilled problem statements, root causes, and resolutions. The cleaner your retained memory records, the sharper your recall queries will be.
3. **Targeted recall queries outperform passive lookups**: Do not simply search for the exact text of the new ticket. Formulate explicit diagnostic questions during recall (e.g., *"Did this user report network partition errors in the past 7 days?"*).
4. **Memory is the difference between a toy and a teammate**: A support agent that forgets past context after closing a browser tab is just an interactive FAQ search. Memory turns an agent into a persistent collaborator that understands operational history.

Building reliable AI agents is not about making prompts longer; it is about giving the system the right memory primitives so it can make decisions with full historical context.
