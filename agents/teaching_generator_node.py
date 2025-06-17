import logging
import json
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from state import AgentGraphState

logger = logging.getLogger(__name__)

def format_rag_content_for_prompt(rag_data: dict) -> str:
    if not rag_data:
        return "No specific pre-defined content was retrieved for this lesson segment."
    
    formatted_rag_parts = []
    for key, value in rag_data.items():
        if isinstance(value, str) and value.strip():
            formatted_rag_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
        elif isinstance(value, list) and value:
            formatted_rag_parts.append(f"- {key.replace('_', ' ').title()}: {'; '.join(map(str,value))}")
            
    if not formatted_rag_parts:
        return "Retrieved pre-defined content was minimal or empty."
        
    return "\nRelevant Pre-defined Content Snippets (for inspiration/guidance):\n" + "\n".join(formatted_rag_parts)

async def teaching_generator_node(state: AgentGraphState) -> dict:
    logger.info("TeachingGeneratorNode: Processing started.")

    # Configure Google Generative AI and initialize model
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("TeachingGeneratorNode: GOOGLE_API_KEY environment variable is not set.")
        return {
            "teaching_output_content": {
                "text_for_tts": "Error: GOOGLE_API_KEY not set. Teaching model unavailable.",
                "ui_actions": []
            }
        }
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        )
        logger.info("TeachingGeneratorNode: Gemini model configured successfully.")
    except Exception as e:
        logger.error(f"TeachingGeneratorNode: Error initializing Gemini model: {e}", exc_info=True)
        return {
            "teaching_output_content": {
                "text_for_tts": "Error: Failed to initialize teaching model.",
                "ui_actions": []
            }
        }

    current_context = state.get("current_context")
    if not current_context:
        logger.error("TeachingGeneratorNode: 'current_context' is missing.")
        return {"teaching_output_content": {"text_for_tts": "Error: Missing context.", "ui_actions": []}, "error": "Missing current_context"}

    retrieved_teaching_row = state.get("retrieved_teaching_row") or {}
    rag_context_for_llm = format_rag_content_for_prompt(retrieved_teaching_row)

    # --- Gather all necessary context variables ---
    teacher_persona = current_context.teacher_persona or "Default TOEFL Tutor"
    learning_objective_id = current_context.learning_objective_id or "General Speaking Practice"
    # Potentially fetch LO description from retrieved_teaching_row or a dedicated source if available
    learning_objective_description = retrieved_teaching_row.get("LEARNING_OBJECTIVE", learning_objective_id) 
    student_proficiency_level = current_context.student_proficiency_level or "Intermediate"
    current_student_affective_state = current_context.current_student_affective_state or "Neutral"
    current_lesson_step_number = current_context.current_lesson_step_number or 1
    is_first_step = current_lesson_step_number == 1
    # is_last_step = ... (this would require knowing total steps for the LO)

    # Context similar to modelling_generator_node
    student_goal_context = state.get("student_goal_context", "improve general speaking skills for TOEFL")
    student_confidence_context = state.get("student_confidence_context", "average")
    student_struggle_context = state.get("student_struggle_context", "occasional hesitation and finding precise vocabulary")
    teacher_initial_impression = state.get("teacher_initial_impression", "seems motivated but a bit unsure")
    english_comfort_level = state.get("english_comfort_level", "fairly comfortable but not native")

    # --- Construct the LLM Prompt ---
    # We want the LLM to output a JSON object with the specified keys.
    prompt = f"""
Act as '{teacher_persona}', an expert AI TOEFL Tutor. Your current task is to generate a script for a teaching lesson segment.

**Student Profile & Context:**
- Learning Objective: {learning_objective_description} (ID: {learning_objective_id})
- Current Lesson Step: {current_lesson_step_number}
- Student's Proficiency: {student_proficiency_level}
- Student's Current Affective State: {current_student_affective_state}
- Student's Goal: {student_goal_context}
- Student's Confidence: {student_confidence_context}
- Student's English Comfort Level: {english_comfort_level}
- Teacher's Initial Impression of Student: {teacher_initial_impression}
- Student's Primary Struggle: {student_struggle_context}

**Guidance from Pre-defined Materials (use as strong inspiration and ensure alignment, but generate fresh, coherent text):**
{rag_context_for_llm}

**Your Task:**
Generate the teaching content for this specific lesson segment. Ensure your language is appropriate for the student's proficiency and addresses their affective state. 

**Output Format:**
Please provide your response as a single JSON object matching the structure below. 
For the 'visual_aid_suggestion' field, if a visual aid is needed, populate the JSON object with the required details. If no visual aid is needed, provide an empty JSON object: {{}}.

Important: For 'write' and 'drawSVG' commands within 'visual_aid_suggestion.steps.payload', the 'position' field (e.g., {{"x": ..., "y": ...}}) is optional. If you are unsure about a non-overlapping position, **omit the 'position' field entirely** from the payload. The frontend system will then attempt to find a suitable empty spot.

{{
  "lesson_opening_hook": "(String: An engaging sentence or two to start this segment, especially if it's step 1. If not step 1, this can be a brief transition. Keep it concise.)",
  "affective_adaptation_narrative": "(String: A brief, empathetic narrative acknowledging the student's affective state and how you'll support them. Integrate this naturally into the lesson's flow.)",
  "core_explanation": "(String: Clear and concise explanation of the key concept for this lesson step, tailored to the learning objective and student proficiency.)",
  "key_examples": "(String: One or two clear and relevant examples illustrating the core explanation. Format them clearly.)",
  "visual_aid_suggestion": {{  // This is an EXAMPLE, adapt it to the current lesson content or provide an empty {{}} if no visual is needed.
    "lessonTitle": "The Pythagorean Theorem",
    "canvasDimensions": {{ "width": 1000, "height": 750 }},
    "steps": [
      {{
        "command": "write",
        "id": "title-text",
        "payload": {{
          "text": "The Pythagorean Theorem",
          "position": {{ "x": 50, "y": 50 }},
          "varaOptions": {{ "fontSize": 48, "color": "#333333", "duration": 2000 }}
        }}
      }},
      {{
        "command": "wait",
        "payload": {{ "duration": 1000 }}
      }},
      {{
        "command": "drawShape",
        "id": "triangle-side-a",
        "payload": {{
          "shapeType": "line",
          "points": [
            {{ "x": 150, "y": 450 }},
            {{ "x": 550, "y": 450 }}
          ],
          "isRough": true,
          "roughOptions": {{
            "stroke": "black",
            "strokeWidth": 4,
            "roughness": 1.5
          }}
        }}
      }},
      {{
        "command": "drawShape",
        "id": "triangle-side-b",
        "payload": {{
          "shapeType": "line",
          "points": [
            {{ "x": 150, "y": 450 }},
            {{ "x": 150, "y": 200 }}
          ],
          "isRough": true,
          "roughOptions": {{
            "stroke": "black",
            "strokeWidth": 4,
            "roughness": 1.5
          }}
        }}
      }},
      {{
        "command": "wait",
        "payload": {{ "duration": 500 }}
      }},
      {{
        "command": "drawShape",
        "id": "triangle-hypotenuse",
        "payload": {{
          "shapeType": "line",
          "points": [
            {{ "x": 150, "y": 200 }},
            {{ "x": 550, "y": 450 }}
          ],
          "isRough": true,
          "roughOptions": {{
            "stroke": "darkblue",
            "strokeWidth": 4,
            "roughness": 1
          }}
        }}
      }},
      {{
        "command": "write",
        "id": "label-a",
        "payload": {{
          "text": "a",
          "position": {{ "x": 350, "y": 470 }},
          "varaOptions": {{ "fontSize": 36, "color": "#555" }}
        }}
      }},
      {{
        "command": "write",
        "id": "label-b",
        "payload": {{
          "text": "b",
          "position": {{ "x": 100, "y": 325 }},
          "varaOptions": {{ "fontSize": 36, "color": "#555" }}
        }}
      }},
      {{
        "command": "write",
        "id": "label-c",
        "payload": {{
          "text": "c",
          "position": {{ "x": 370, "y": 300 }},
          "varaOptions": {{ "fontSize": 36, "color": "darkblue" }}
        }}
      }},
      {{
        "command": "write",
        "id": "formula-text",
        "payload": {{
          "text": "a² + b² = c²",
          "position": {{ "x": 600, "y": 150 }},
          "varaOptions": {{ "fontSize": 52, "color": "#d9534f", "duration": 2500 }}
        }}
      }},
      {{
        "command": "wait",
        "payload": {{ "duration": 1500 }}
      }},
      {{
        "command": "annotate",
        "id": "formula-annotation",
        "payload": {{
          "targetId": "formula-text",
          "annotationType": "circle",
          "isRough": true,
          "roughOptions": {{
            "stroke": "#d9534f",
            "strokeWidth": 3,
            "roughness": 2.5
          }}
        }}
      }},
      {{
        "command": "wait",
        "payload": {{ "duration": 1000 }}
      }},
      {{
        "command": "drawSVG",
        "id": "ladder-example-svg",
        "payload": {{
          "svgUrl": "https://raw.githubusercontent.com/google/material-design-icons/master/png/action/build_circle/materialicons/48dp/2x/baseline_build_circle_black_48dp.png",
          "description": "An SVG diagram showing a ladder against a wall, forming a right triangle.",
          "position": {{ "x": 450, "y": 550 }},
          "desiredSize": {{ "width": 200 }}
        }}
      }},
      {{
        "command": "write",
        "id": "example-explanation",
        "payload": {{
          "text": "It's used to find unknown lengths in the real world!",
          "position": {{ "x": 650, "y": 600 }},
          "varaOptions": {{ "fontSize": 24, "color": "#0275d8" }}
        }}
      }}
    ]
  }},
  "common_misconceptions_addressed": "(String: Briefly mention and clarify any common misconceptions related to this step's topic.)",
  "comprehension_check_question": "(String: A short, open-ended question to check the student's understanding of this segment's content.)",
  "lesson_summary_for_step": "(String: A very brief (1-2 sentence) summary of what was covered in this specific step. If this is the final step of an overall lesson for the LO, make this a slightly broader summary of the LO.)"
}}

Ensure the entire output is a single, valid JSON object.
"""

    logger.debug(f"TeachingGeneratorNode: Prompt for LLM:\n{prompt}")

    try:
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.7, # Allow for some creativity in teaching style
            top_p=0.9,
            top_k=40
        )
        response = await model.generate_content_async(prompt, generation_config=generation_config, safety_settings={ # Adjust safety settings as needed
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        logger.info("TeachingGeneratorNode: Received response from LLM.")
        llm_output_text = response.text
        logger.info(f"TeachingGeneratorNode: LLM Raw Response (first 500 chars): {llm_output_text[:500]}")
        
        # Attempt to parse the JSON output
        try:
            generated_content = json.loads(llm_output_text)
        except json.JSONDecodeError as e:
            logger.error(f"TeachingGeneratorNode: Failed to decode LLM JSON response. Error: {e}. Response text: {llm_output_text}")
            # Fallback: Try to extract content if JSON is malformed, or use a default error message
            # For now, returning an error message in TTS
            return {
                "teaching_output_content": {
                    "text_for_tts": "I had a thought, but I'm having a little trouble expressing it clearly right now. Let's try that again in a moment.", 
                    "ui_actions": []
                },
                "error": "LLM JSON parsing error"
            }

        # --- Assemble text_for_tts and ui_actions from parsed LLM response ---
        tts_parts = []
        if is_first_step and generated_content.get("lesson_opening_hook"):
            tts_parts.append(generated_content["lesson_opening_hook"])
        
        if generated_content.get("affective_adaptation_narrative"):
            tts_parts.append(generated_content["affective_adaptation_narrative"])
        
        if generated_content.get("core_explanation"):
            tts_parts.append(generated_content["core_explanation"])
        
        if generated_content.get("key_examples"):
            # Could add a prefix like "For example:"
            tts_parts.append(f"For instance: {generated_content['key_examples']}")
            
        if generated_content.get("common_misconceptions_addressed"):
            tts_parts.append(f"A common point of confusion is: {generated_content['common_misconceptions_addressed']}")

        if generated_content.get("comprehension_check_question"):
            tts_parts.append(f"To check your understanding: {generated_content['comprehension_check_question']}")

        if generated_content.get("lesson_summary_for_step"):
            tts_parts.append(generated_content["lesson_summary_for_step"])

        final_tts = " ".join(filter(None, tts_parts)).strip()
        if not final_tts:
            final_tts = "I'm ready to continue when you are."
            logger.warning("TeachingGeneratorNode: LLM generated empty or minimal TTS content.")

        ui_actions = []
        visual_suggestion = generated_content.get("visual_aid_suggestion")
        # Check if visual_suggestion is a non-empty dictionary. An empty {} means no visual.
        if visual_suggestion and isinstance(visual_suggestion, dict):
            ui_actions.append({
                "action_type": "DISPLAY_VISUAL_AID",
                "visual_description": visual_suggestion, # Assign the object directly, not as a string.
                "details": {
                    "source": "teaching_generator_node_llm",
                    "suggestion_type": "llm_generated_visual_aid"
                }
            })

        logger.info(f"TeachingGeneratorNode: Processing complete. TTS preview (first 100 chars): {final_tts[:100]}. UI Actions count: {len(ui_actions)}")
        return {"teaching_output_content": {"text_for_tts": final_tts, "ui_actions": ui_actions}}

    except Exception as e:
        logger.error(f"TeachingGeneratorNode: Error during LLM call or processing: {e}", exc_info=True)
        return {
            "teaching_output_content": {"text_for_tts": "I encountered a slight hiccup. Could we try that part again?", "ui_actions": []},
            "error": str(e)
        }
