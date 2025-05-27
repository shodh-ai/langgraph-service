import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def provide_scaffolding_node(state: AgentGraphState) -> dict:
    context = state.get("current_context")
    section = getattr(context, "toefl_section", None)
    question_type = getattr(context, "question_type", None)
    diagnosis = state.get("diagnosis_result", {})
    
    logger.info(f"ScaffoldingProviderNode: Providing scaffolding for {section}/{question_type}")
    
    scaffolding = {
        "templates": [],
        "hints": [],
        "task_breakdown": []
    }
    
    # speaking scaffolding
    if section == "Speaking":
        if question_type == "Q1_Independent":
            scaffolding["templates"] = [
                "In my opinion, [state position]...",
                "I believe that [state position] because [reason 1] and [reason 2]..."
            ]
            scaffolding["task_breakdown"] = [
                "1. Introduce your position (5-10 seconds)",
                "2. State first reason with example (20-25 seconds)",
                "3. State second reason with example (20-25 seconds)",
                "4. Conclude (5 seconds)"
            ]
    
    # writing scaffolding
    elif section == "Writing":
        if question_type == "Integrated_Essay":
            scaffolding["templates"] = [
                "The reading passage discusses [main topic]. The lecture supports/challenges this by...",
                "According to the reading, [key point 1]. However, the lecturer argues that..."
            ]
            scaffolding["task_breakdown"] = [
                "1. Introduction: Summarize the relationship between reading and lecture (1 paragraph)",
                "2. Body Paragraph 1: First point from reading and corresponding lecture point (1 paragraph)",
                "3. Body Paragraph 2: Second point from reading and corresponding lecture point (1 paragraph)",
                "4. Body Paragraph 3: Third point from reading and corresponding lecture point (1 paragraph)",
                "5. Conclusion: Optional brief summary (1 short paragraph)"
            ]
    
    if diagnosis.get("primary_error") == "limited vocabulary range":
        scaffolding["hints"].append("Try using synonyms for repeated words. For example, instead of always saying 'good', consider: excellent, beneficial, valuable, advantageous.")
    elif diagnosis.get("primary_error") == "some grammatical inconsistencies":
        scaffolding["hints"].append("Check your verb tenses to ensure consistency. If you start in past tense, maintain it unless there's a reason to switch.")
    
    logger.info(f"ScaffoldingProviderNode: Completed scaffolding generation")
    
    return {"scaffolding_content": scaffolding}