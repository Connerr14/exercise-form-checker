class SquatCoach:
    def __init__(self):
        self.reset_rep()
        self.rep_count = 0
        self.last_verdict = ""
        self.live_feedback = "Ready"

    def reset_rep(self):
        self.has_hit_bottom = False
        self.errors_found = set()
        self.in_rep = False

    def process_frame(self, stable_label, raw_label):
        """
        Modified to take BOTH the stable (for UI) and raw (for depth) labels.
        """
        # --- A. START TRIGGER ---
        if stable_label != "Standing" and not self.in_rep:
            self.in_rep = True
            self.last_verdict = "" 
            self.live_feedback = "MOVEMENT DETECTED..."

        if self.in_rep:
            # --- B. SENSITIVE DEPTH CHECK ---
            # We check the RAW label here. If the AI sees the bottom 
            # for even one frame, we count it!
            if raw_label == "At_Bottom" or stable_label == "At_Bottom":
                self.has_hit_bottom = True
                self.live_feedback = "DEPTH REACHED! UP!"

            # --- C. FEEDBACK UPDATES ---
            if stable_label == "Descending" and not self.has_hit_bottom:
                self.live_feedback = "GO LOWER..."
            
            elif stable_label == "Ascending":
                self.live_feedback = "DRIVING UP..."

            # Error Logging
            if stable_label in ["Uneven_Weight", "Forward_Lean", "Force_Imbalance"]:
                self.errors_found.add(stable_label)
                self.live_feedback = f"FIX: {stable_label.replace('_', ' ').upper()}"

            # --- D. FINISH ---
            if stable_label == "Standing":
                verdict, should_count = self.evaluate_rep()
                if should_count:
                    self.rep_count += 1
                self.last_verdict = verdict
                self.reset_rep()
                return verdict
        else:
            self.live_feedback = "READY - START SQUAT"
    
    def evaluate_rep(self):
            # 1. If they didn't hit depth, it's NOT a rep (No count)
            if not self.has_hit_bottom:
                return "TOO SHALLOW", False
            
            # 2. If they have errors, it IS a rep, but with a warning
            if self.errors_found:
                errors_str = ", ".join(self.errors_found)
                return f"REP OK (Fix: {errors_str})", True
                
            # 3. Perfect rep
            return "PERFECT REP", True