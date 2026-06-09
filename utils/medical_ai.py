import os
import json
import urllib.request

class MedicalAIEngine:
    def __init__(self, high_threshold=70.0, intermediate_threshold=50.0):
        self.high_threshold = high_threshold
        self.intermediate_threshold = intermediate_threshold
        self.vector_store_client = None  # Future integration spot (e.g. ChromaDB or Pinecone)

    def retrieve_context(self, query):
        """
        RAG Document Retrieval Interface.
        Currently queries our comprehensive local medical knowledge base.
        In the future, swap this method with a vector database similarity search 
        (e.g., ChromaDB/Pinecone client query) without breaking any caller signatures.
        """
        if self.vector_store_client:
            # Future Pinecone/Chroma search:
            # return self.vector_store_client.similarity_search(query, k=3)
            pass
        return self._local_document_search(query)

    def _local_document_search(self, query):
        """
        Retrieves matching paragraphs from our built-in local uveal melanoma knowledge base.
        """
        query_clean = query.strip().lower()
        matches = []
        
        # Local document chunks
        knowledge_base = [
            {
                "topic": "definition",
                "content": "Uveal Melanoma is a primary intraocular malignancy of the eye developing in melanocytes of the uveal tract (iris, ciliary body, or choroid). It is the most common primary eye cancer in adults. While rare, it carries a high risk of systemic metastasis, specifically hematogenous spread to the liver in ~90% of metastatic cases."
            },
            {
                "topic": "risk factors",
                "content": "Host risk factors for uveal melanoma include fair skin, light eye color (blue, green, gray), ocular melanocytosis (Nevus of Ota), atypical cutaneous nevi, and genetic predisposition. Incidence peaks in the 60-70 age range and is most prevalent in Caucasian populations."
            },
            {
                "topic": "diagnosis",
                "content": "Diagnosis of uveal melanoma is primarily clinical and does not typically require biopsy. Methods include ophthalmoscopy, high-frequency ocular B-scan ultrasonography (dome or mushroom mass with low internal reflectivity), fluorescein angiography (double circulation), and optical coherence tomography (OCT)."
            },
            {
                "topic": "treatment",
                "content": "Treatments are categorized into eye-preserving therapies and surgery. Radioactive plaque brachytherapy (Iodine-125 or Ruthenium-106) is the gold standard for small-to-medium tumors. Proton beam radiotherapy is also utilized. Enucleation (surgical excision of the globe) is reserved for large, painful, or ciliary-body invasive tumors."
            },
            {
                "topic": "prognosis and genetics",
                "content": "Prognosis depends on clinical features (AJCC Stage, tumor diameter, and ciliary body involvement) and genetic markers. Loss of Chromosome 3 (Monosomy 3) and BAP1 mutations are associated with a very high metastatic risk, while Chromosome 8q gain indicates poor prognosis."
            },
            {
                "topic": "model details",
                "content": "This Prognostic System uses an XGBoost machine learning model trained on SEER population-level historical data. It evaluates Sex, Age Range, Race, derived AJCC Stage, Radiation Therapy, and Chemotherapy to compute the statistical likelihood of 5-year metastasis survival."
            }
        ]
        
        for doc in knowledge_base:
            # Simple keyword matching for context retrieval
            if any(word in query_clean for word in doc["topic"].split()):
                matches.append(doc["content"])
        
        if not matches:
            # Fallback default knowledge context
            matches.append(knowledge_base[0]["content"])
            
        return "\n\n".join(matches)

    def safety_scan(self, message):
        """
        Scans input for clinical directives and returns standard disclaimer if triggered.
        Refuses diagnosis, prescriptions, emergency advice, prognosis prediction, or patient-specific decisions.
        """
        import re
        msg_lower = message.lower()
        
        # Word boundary triggers
        word_triggers = [
            r"\bprescribe\b", r"\bdiagnose\b", r"\bemergency\b", r"\bprescribed\b", r"\bprescribing\b", r"\bdiagnosing\b"
        ]
        # Phrase triggers (substring matches are fine)
        phrase_triggers = [
            "recipe", "what drug", "what medicine", "what treatment should i", 
            "give me a prescription", "should i get surgery", "is this eye cancer", 
            "do i have cancer", "treat me", "clinical advice", "recommend a treatment", 
            "treatment for my", "treatment for me", "my treatment", "treatment recommendation", 
            "cure my", "cure me", "prognosis prediction", "prognosis for me", 
            "my prognosis", "patient-specific"
        ]
        
        # Check word triggers
        for pattern in word_triggers:
            if re.search(pattern, msg_lower):
                return (
                    "This AI Medical Assistant provides educational information only and is not a substitute for "
                    "professional medical evaluation. Please consult a qualified ophthalmologist or healthcare professional."
                )
                
        # Check phrase triggers
        for trigger in phrase_triggers:
            if trigger in msg_lower:
                return (
                    "This AI Medical Assistant provides educational information only and is not a substitute for "
                    "professional medical evaluation. Please consult a qualified ophthalmologist or healthcare professional."
                )
                
        return None



    def query(self, message, history=None, patient_data=None):
        """
        Core query execution. Resolves safety, reads config thresholds, parses SHAP explainability, 
        and calls Groq LLM (or fallback simulation).
        """
        # 1. Immediate Safety Verification
        safety_response = self.safety_scan(message)
        if safety_response:
            return safety_response

        # 2. Extract retrieve context for RAG
        retrieved_context = self.retrieve_context(message)

        # 3. Handle Explainability Support (SHAP context extraction)
        explainability_prompt = ""
        if patient_data:
            # Extract inputs
            age = patient_data.get("age", "Unknown")
            sex = "Female" if str(patient_data.get("sex")) == "1" else "Male"
            stage = patient_data.get("stage", "Unknown")
            risk_score = patient_data.get("risk_score", 0.0)
            risk_category = patient_data.get("risk_category", "Unknown")
            shap_details = patient_data.get("shap_details")

            # Check if SHAP details are present
            if shap_details and isinstance(shap_details, dict):
                # Sort features by absolute contribution (highest impact first)
                sorted_features = sorted(shap_details.items(), key=lambda x: abs(x[1]), reverse=True)
                
                explainability_prompt = (
                    f"PREDICTION EXPLANATION CONTEXT:\n"
                    f"- Current Patient Details: Sex={sex}, Age={age}, Stage={stage}\n"
                    f"- Model Prediction Output: Risk Score={risk_score}%, Risk Category={risk_category}\n"
                    f"- Actual SHAP Contribution Scores (Ordered by Impact):\n"
                )
                for feat, val in sorted_features:
                    impact_direction = "increases risk" if val > 0 else "decreases risk (protective)"
                    explainability_prompt += f"  * {feat}: {val} ({impact_direction})\n"
                
                explainability_prompt += (
                    "\nINSTRUCTION:\n"
                    "The user is in Prediction Explanation Mode. You MUST prioritize explaining the actual model-driving features "
                    "listed in the SHAP contribution scores above. Do not give generic explanations. Clearly state which features "
                    "had the largest positive or negative impact on the calculated risk score for this specific patient."
                )
            else:
                # Fallback to patient inputs explanation if SHAP is missing
                explainability_prompt = (
                    f"PREDICTION EXPLANATION CONTEXT (SHAP details unavailable):\n"
                    f"- Current Patient Details: Sex={sex}, Age={age}, Stage={stage}\n"
                    f"- Model Prediction Output: Risk Score={risk_score}%, Risk Category={risk_category}\n"
                    f"\nINSTRUCTION:\n"
                    f"Explain the prediction using the available patient input parameters listed above. Keep it educational."
                )

        # 4. Define system prompt dynamically injecting config thresholds
        system_prompt = (
            f"You are a highly specialized AI Medical Assistant for the Uveal Melanoma Prognostic System.\n"
            f"Your role is to help doctors, researchers, and users understand uveal melanoma and prognostic output.\n\n"
            f"MODEL CONFIGURATION DETAILS:\n"
            f"- High Risk threshold: > {self.high_threshold}%\n"
            f"- Intermediate Risk threshold: {self.intermediate_threshold}% to {self.high_threshold}%\n"
            f"- Low Risk threshold: < {self.intermediate_threshold}%\n\n"
            f"RETRIEVED MEDICAL CONTEXT:\n"
            f"{retrieved_context}\n\n"
            f"{explainability_prompt}\n\n"
            f"SAFETY PROTOCOL:\n"
            f"For diagnoses, prescriptions, emergency advice, prognosis predictions, treatment recommendations, or patient-specific "
            f"medical decisions, you MUST provide an educational disclaimer and redirect the user to a qualified clinical professional. "
            f"Emergency responses must be redirected to emergency services.\n"
        )

        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            try:
                # Call Groq chat completions using Llama 3.3 70B
                url = "https://api.groq.com/openai/v1/chat/completions"
                messages = [{"role": "system", "content": system_prompt}]
                
                # Append history
                if history:
                    for h in history[-6:]:
                        messages.append({"role": h.get("role"), "content": h.get("content")})
                
                messages.append({"role": "user", "content": message})
                
                data = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 1024
                }
                
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    },
                    method="POST"
                )
                
                with urllib.request.urlopen(req, timeout=12) as response:
                    res_json = json.loads(response.read().decode("utf-8"))
                    return res_json["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Groq API Error: {e}. Falling back to simulation mode.")
                pass

        # 5. Local simulation fallback (reads retrieved context and explainability)
        return self._simulate_response(message, retrieved_context, explainability_prompt)

    def _simulate_response(self, query, context, explainability):
        """
        Clinical rule-based response generator when Groq is unavailable.
        Uses SHAP details and context to formulate precise, non-generic responses.
        """
        clean_query = query.strip().lower()

        # Handle prediction explanation request
        if explainability:
            # Parse explainability variables
            is_high = "Risk Category=High" in explainability
            is_intermediate = "Risk Category=Intermediate" in explainability
            
            import re
            risk_score_match = re.search(r"Risk Score=([\d\.]+)", explainability)
            risk_score_val = risk_score_match.group(1) if risk_score_match else "Unknown"
            
            # Simple regex/split simulation to extract SHAP lines
            shap_lines = [line.strip() for line in explainability.split('\n') if "  *" in line]
            
            explanation = (
                f"### AI Explanation of Patient Prediction Result\n\n"
                f"Based on the **XGBoost Prognostic Model** configuration (High Risk: >{self.high_threshold}%, "
                f"Intermediate: {self.intermediate_threshold}%-{self.high_threshold}%), this patient is classified as **"
                f"{'High Risk' if is_high else ('Intermediate Risk' if is_intermediate else 'Low Risk')}** "
                f"with a calculated prognostic risk score of **{risk_score_val}%**.\n\n"
                f"#### Model Driver Breakdown (SHAP Impact Analysis):\n"
            )
            
            if shap_lines:
                for line in shap_lines[:3]:  # Top 3 drivers
                    # Clean line from list asterisk
                    clean_line = line.replace('*', '').strip()
                    explanation += f"* **{clean_line}**\n"
                explanation += (
                    "\n#### Clinical Interpretation:\n"
                    "The primary risk contributors have been identified above. These feature impacts denote "
                    "how much the patient's specific clinical features pushed the output risk probability higher "
                    "or lower relative to the baseline population average. These estimates are derived from historical SEER registry data."
                )
            else:
                explanation += (
                    "The patient features have been evaluated against our historical cohort. Detailed SHAP impact "
                    "coefficients were not found, but risk is estimated based on the clinical stage group and demographics."
                )
            
            explanation += (
                "\n\n*This AI Medical Assistant provides educational information only and is not a substitute for "
                "professional medical evaluation. Please consult a qualified ophthalmologist or healthcare professional.*"
            )
            return explanation

        # Match general queries
        if "what is uveal melanoma" in clean_query:
            return (
                "### Uveal Melanoma Overview\n\n"
                f"{context}\n\n"
                "Symptoms typically include blurred vision, floaters, flashing lights, or a growing dark spot on the iris. "
                "However, many patients have no early symptoms, and the tumor is often detected during a routine eye exam."
            )
            
        if "risk factors" in clean_query or "causes" in clean_query:
            return (
                "### Risk Factors for Uveal Melanoma\n\n"
                f"{context}\n\n"
                "It is critical to note that while skin melanoma is highly linked to UV exposure, the association between "
                "sunlight/UV exposure and uveal melanoma is still debated and much weaker. Genetic predisposition (such as BAP1 tumor "
                "predisposition syndrome) plays a major role."
            )

        if "diagnostic" in clean_query or "diagnos" in clean_query:
            return (
                "### Diagnosis Methods\n\n"
                f"{context}\n\n"
                "Ultrasound B-scan remains a primary clinical diagnostic method, revealing characteristic dome or mushroom shapes, "
                "regular internal structure, and choroidal excavation. Fluorescein angiography can show double circulation (intrinsic "
                "tumor vessels and retinal vessels) confirming malignancy."
            )

        if "treatment" in clean_query or "radiation" in clean_query or "surgery" in clean_query:
            return (
                "### Treatment Options Overview\n\n"
                f"{context}\n\n"
                "Radiotherapy (plaques or proton beam) yields local control rates exceeding 95%. However, eye-preserving treatments "
                "do not reduce the long-term risk of distant metastasis compared to enucleation. Systematic follow-ups (liver MRI, "
                "ultrasound, or LFTs) are recommended for early detection of systemic metastasis."
            )

        if "limitations" in clean_query or "limit" in clean_query:
            return (
                "### Model and Supportive Limitations\n\n"
                f"{context}\n\n"
                "Please remember that this prognostic tool evaluates generic clinical statistics from historical SEER registries. "
                "It is not capable of molecular prognostication (e.g. chromosome 3 status or gene expression profiling like DecisionDx-UM), "
                "which are now standard in advanced ocular oncology clinics."
            )

        # Default query template response
        return (
            f"**AI Medical Assistant (Simulation Mode)**\n\n"
            f"I have retrieved the following educational context regarding your query:\n"
            f"> {context}\n\n"
            f"To get patient-specific analysis, run a calculation under **Prediction Model** and click **'Explain with AI Medical Assistant'**."
        )
