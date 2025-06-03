import logging
from state import AgentGraphState
import yaml
import os
import json
# Try to import vertexai, but don't fail if it's not available
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Content, Part
    vertexai_available = True
except ImportError:
    vertexai_available = False
    
# Import our fallback utilities
from utils.fallback_utils import get_model_with_fallback

logger = logging.getLogger(__name__)
PROMPTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "llm_prompts.yaml")

try:
    with open(PROMPTS_PATH, 'r') as file:
        PROMPTS = yaml.safe_load(file)
except Exception as e:
    logger.error(f"Failed to load LLM prompts: {e}")
    PROMPTS = {}

# Use our fallback utility to get a model (either real or fallback)
gemini_model = get_model_with_fallback("gemini-2.5-flash-preview-05-20")
logger.info(f"Using model: {type(gemini_model).__name__} in scaffolding_provider_node")

async def provide_scaffolding_node(state: AgentGraphState) -> dict:
    context = state.get("current_context")
    section = getattr(context, "toefl_section", None)
    question_type = getattr(context, "question_type", None)
    diagnosis = state.get("diagnosis_result", {})
    
    logger.info(f"ScaffoldingProviderNode: Providing scaffolding for {section}/{question_type}")
    
    if section == "Speaking":
        scaffolding = await provide_scaffolding_for_speaking_node(state)
    elif section == "Writing":
        scaffolding = await provide_scaffolding_for_writing_node(state)
    else:
        logger.warning(f"ScaffoldingProviderNode: Unsupported section: {section}")
        scaffolding = {}
    
    logger.info(f"ScaffoldingProviderNode: Completed scaffolding generation")
    
    return {"scaffolding_content": scaffolding}

async def provide_scaffolding_for_speaking_node(state: AgentGraphState) -> dict:
    """Provides templates, hints, and task breakdowns for speaking tasks using Vertex AI Gemini."""
    task_prompt = state.get("task_prompt", {})
    diagnosis = state.get("diagnosis_result", {})
    student_data = state.get("student_memory_context", {})
    
    logger.info(f"ScaffoldingProviderNode: Generating speaking scaffolding")
    
    if not gemini_model:
        logger.warning("ScaffoldingProviderNode: Gemini model not available, using stub implementation for speaking")
        scaffolding = {
            "template": "I believe [opinion] because [reason 1], [reason 2], and [reason 3]. For example, [specific example].",
            "hints": [
                "Start with a clear statement of your position",
                "Include at least 2-3 supporting reasons",
                "Provide a specific example to illustrate your point"
            ],
            "breakdown": [
                "Introduction (10-15 seconds): State your opinion clearly",
                "Body (30-40 seconds): Explain 2-3 reasons with brief examples",
                "Conclusion (5-10 seconds): Restate your opinion or provide a final thought"
            ]
        }
    else:
        try:
            # Get prompt templates from config
            system_prompt = PROMPTS.get("scaffolding", {}).get("speaking", {}).get("system", "")
            user_prompt = PROMPTS.get("scaffolding", {}).get("speaking", {}).get("user_template", "")
            
            # Replace placeholders in prompts
            system_prompt = system_prompt.replace("{{task_prompt}}", json.dumps(task_prompt))
            system_prompt = system_prompt.replace("{{diagnosis_result}}", json.dumps(diagnosis))
            system_prompt = system_prompt.replace("{{student_memory_context}}", json.dumps(student_data))
            
            # Create Gemini content
            contents = [
                Content(role="user", parts=[system_prompt]),
                Content(role="model", parts=["I understand. I'll generate scaffolding for a speaking task based on the student's needs and the task requirements."]),
                Content(role="user", parts=[user_prompt])
            ]
            
            # Generate response from Gemini
            response = gemini_model.generate_content(contents, generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1024
            })
            
            # Try to parse the response as JSON
            response_text = response.text
            try:
                scaffolding = json.loads(response_text)
                logger.info("ScaffoldingProviderNode: Successfully parsed JSON scaffolding for speaking")
            except json.JSONDecodeError:
                logger.warning(f"ScaffoldingProviderNode: Failed to parse JSON response for speaking: {response_text}")
                template = ""
                hints = []
                breakdown = []
                
                # Simple parsing logic - can be improved
                lines = response_text.split("\n")
                section = None
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if "template" in line.lower() or "structure" in line.lower():
                        section = "template"
                        template_text = line.split(":", 1)[1].strip() if ":" in line else ""
                        if template_text:
                            template = template_text
                        continue
                    
                    if "hint" in line.lower() or "tip" in line.lower():
                        section = "hints"
                        continue
                    
                    if "breakdown" in line.lower() or "step" in line.lower() or "structure" in line.lower():
                        section = "breakdown"
                        continue
                    
                    if section == "template" and not template:
                        template = line
                    elif section == "hints" and line.strip("-*").strip():
                        hints.append(line.strip("-*").strip())
                    elif section == "breakdown" and line.strip("-*").strip():
                        breakdown.append(line.strip("-*").strip())
                
                # Ensure we have at least some content
                if not template:
                    template = "I believe [opinion] because [reason 1], [reason 2], and [reason 3]. For example, [specific example]."
                
                if not hints:
                    hints = [
                        "Start with a clear statement of your position",
                        "Include at least 2-3 supporting reasons",
                        "Provide a specific example to illustrate your point"
                    ]
                
                if not breakdown:
                    breakdown = [
                        "Introduction (10-15 seconds): State your opinion clearly",
                        "Body (30-40 seconds): Explain 2-3 reasons with brief examples",
                        "Conclusion (5-10 seconds): Restate your opinion or provide a final thought"
                    ]
                
                scaffolding = {
                    "template": template,
                    "hints": hints,
                    "breakdown": breakdown
                }
                
        except Exception as e:
            logger.error(f"ScaffoldingProviderNode: Error calling Gemini API for speaking: {e}")
            # Fallback to stub implementation
            scaffolding = {
                "template": "I believe [opinion] because [reason 1], [reason 2], and [reason 3]. For example, [specific example].",
                "hints": [
                    "Start with a clear statement of your position",
                    "Include at least 2-3 supporting reasons",
                    "Provide a specific example to illustrate your point"
                ],
                "breakdown": [
                    "Introduction (10-15 seconds): State your opinion clearly",
                    "Body (30-40 seconds): Explain 2-3 reasons with brief examples",
                    "Conclusion (5-10 seconds): Restate your opinion or provide a final thought"
                ]
            }
    
    logger.info(f"ScaffoldingProviderNode: Generated speaking scaffolding with {len(scaffolding['hints'])} hints")
    
    return {"speaking_scaffolding": scaffolding}

async def provide_scaffolding_for_writing_node(state: AgentGraphState) -> dict:
    task_prompt = state.get("task_prompt", {})
    diagnosis = state.get("diagnosis_result", {})
    student_data = state.get("student_memory_context", {})
    
    logger.info(f"ScaffoldingProviderNode: Generating writing scaffolding")
    
    if not gemini_model:
        logger.warning("ScaffoldingProviderNode: Gemini model not available, using stub implementation for writing")
        # Fallback to stub implementation
        scaffolding = {
            "template": "In my opinion, [thesis statement]. First, [main point 1]. For instance, [example 1]. Second, [main point 2]. For example, [example 2]. Finally, [main point 3]. In conclusion, [restate thesis].",
            "hints": [
                "Start with a clear thesis statement",
                "Develop 2-3 main points with examples",
                "Use transition words between paragraphs",
                "Conclude by restating your thesis in different words"
            ],
            "breakdown": [
                "Introduction (1 paragraph): Hook, background, thesis statement",
                "Body Paragraph 1 (1 paragraph): Topic sentence, explanation, example, analysis",
                "Body Paragraph 2 (1 paragraph): Topic sentence, explanation, example, analysis",
                "Body Paragraph 3 (optional): Topic sentence, explanation, example, analysis",
                "Conclusion (1 paragraph): Restate thesis, summarize main points, final thought"
            ]
        }
    else:
        try:
            # Get prompt templates from config
            system_prompt = PROMPTS.get("scaffolding", {}).get("writing", {}).get("system", "")
            user_prompt = PROMPTS.get("scaffolding", {}).get("writing", {}).get("user_template", "")
            
            # Replace placeholders in prompts
            system_prompt = system_prompt.replace("{{task_prompt}}", json.dumps(task_prompt))
            system_prompt = system_prompt.replace("{{diagnosis_result}}", json.dumps(diagnosis))
            system_prompt = system_prompt.replace("{{student_memory_context}}", json.dumps(student_data))
            
            # Create Gemini content
            contents = [
                Content(role="user", parts=[system_prompt]),
                Content(role="model", parts=["I understand. I'll generate scaffolding for a writing task based on the student's needs and the task requirements."]),
                Content(role="user", parts=[user_prompt])
            ]
            
            # Generate response from Gemini
            response = gemini_model.generate_content(contents, generation_config={
                "temperature": 0.3, 
                "max_output_tokens": 1024
            })
            
            # Try to parse the response as JSON
            response_text = response.text
            try:
                scaffolding = json.loads(response_text)
                logger.info("ScaffoldingProviderNode: Successfully parsed JSON scaffolding for writing")
            except json.JSONDecodeError:
                logger.warning(f"ScaffoldingProviderNode: Failed to parse JSON response for writing: {response_text}")
                # Extract scaffolding components from text
                template = ""
                hints = []
                breakdown = []
                
                # Simple parsing logic - can be improved
                lines = response_text.split("\n")
                section = None
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if "template" in line.lower() or "structure" in line.lower():
                        section = "template"
                        template_text = line.split(":", 1)[1].strip() if ":" in line else ""
                        if template_text:
                            template = template_text
                        continue
                    
                    if "hint" in line.lower() or "tip" in line.lower():
                        section = "hints"
                        continue
                    
                    if "breakdown" in line.lower() or "step" in line.lower() or "structure" in line.lower():
                        section = "breakdown"
                        continue
                    
                    if section == "template" and not template:
                        template = line
                    elif section == "hints" and line.strip("-*").strip():
                        hints.append(line.strip("-*").strip())
                    elif section == "breakdown" and line.strip("-*").strip():
                        breakdown.append(line.strip("-*").strip())
                
                # Ensure we have at least some content
                if not template:
                    template = "In my opinion, [thesis statement]. First, [main point 1]. For instance, [example 1]. Second, [main point 2]. For example, [example 2]. Finally, [main point 3]. In conclusion, [restate thesis]."
                
                if not hints:
                    hints = [
                        "Start with a clear thesis statement",
                        "Develop 2-3 main points with examples",
                        "Use transition words between paragraphs",
                        "Conclude by restating your thesis in different words"
                    ]
                
                if not breakdown:
                    breakdown = [
                        "Introduction (1 paragraph): Hook, background, thesis statement",
                        "Body Paragraph 1 (1 paragraph): Topic sentence, explanation, example, analysis",
                        "Body Paragraph 2 (1 paragraph): Topic sentence, explanation, example, analysis",
                        "Body Paragraph 3 (optional): Topic sentence, explanation, example, analysis",
                        "Conclusion (1 paragraph): Restate thesis, summarize main points, final thought"
                    ]
                
                scaffolding = {
                    "template": template,
                    "hints": hints,
                    "breakdown": breakdown
                }
                
        except Exception as e:
            logger.error(f"ScaffoldingProviderNode: Error calling Gemini API for writing: {e}")
            # Fallback to stub implementation
            scaffolding = {
                "template": "In my opinion, [thesis statement]. First, [main point 1]. For instance, [example 1]. Second, [main point 2]. For example, [example 2]. Finally, [main point 3]. In conclusion, [restate thesis].",
                "hints": [
                    "Start with a clear thesis statement",
                    "Develop 2-3 main points with examples",
                    "Use transition words between paragraphs",
                    "Conclude by restating your thesis in different words"
                ],
                "breakdown": [
                    "Introduction (1 paragraph): Hook, background, thesis statement",
                    "Body Paragraph 1 (1 paragraph): Topic sentence, explanation, example, analysis",
                    "Body Paragraph 2 (1 paragraph): Topic sentence, explanation, example, analysis",
                    "Body Paragraph 3 (optional): Topic sentence, explanation, example, analysis",
                    "Conclusion (1 paragraph): Restate thesis, summarize main points, final thought"
                ]
            }
    
    logger.info(f"ScaffoldingProviderNode: Generated writing scaffolding with {len(scaffolding['hints'])} hints")
    
    return {"writing_scaffolding": scaffolding}