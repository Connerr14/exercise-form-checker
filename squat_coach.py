class SquatCoach:
    # The constructor for the class
    def __init__(self):
        self.reset_rep()
        self.rep_count = 0
        self.last_verdict = ""
        self.live_feedback = "Ready"

    # A function to reset the rep data members
    def reset_rep(self):
        self.has_hit_bottom = False
        self.errors_found = set()
        self.in_rep = False

    """A function that processes each frame and updates the cam labels"""
    def process_frame(self, stable_label, raw_label):
        # If the model labeled the frame as not standing, and self.in_rep is not yet set, set it to true
        if stable_label != "Standing" and not self.in_rep:
            self.in_rep = True
            self.last_verdict = "" 
            self.live_feedback = "MOVEMENT DETECTED..."

        # If the user is in the rep, and the model labels the frame "At bottom", set the hasHitBottom to true
        if self.in_rep:
            # Checking the raw label here. If the model sees the bottom for even one frame, count it
            if raw_label == "At_Bottom" or stable_label == "At_Bottom":
                self.has_hit_bottom = True
                self.live_feedback = "DEPTH REACHED! UP!"

            # If the label detected is descending, and the user has not hit the bottom of the squat
            # Tell the user to go lower
            if stable_label == "Descending" and not self.has_hit_bottom:
                self.live_feedback = "GO LOWER..."
            # If the frame is labeled as ascending, set the live feed back as driving up
            elif stable_label == "Ascending":
                self.live_feedback = "DRIVING UP..."

            # If a frame was logged as one of the improper form techniques, add the error to a list
            if stable_label in ["Uneven_Weight", "Forward_Lean", "Force_Imbalance"]:
                self.errors_found.add(stable_label)
                # Inform the user on what needs to be fixed
                self.live_feedback = f"FIX: {stable_label.replace('_', ' ').upper()}"

            # If the frame is labeled as standing, and the user was in the rep, increment the rep counter
            if stable_label == "Standing":
                # Evaluate the rep
                verdict, should_count = self.evaluate_rep()
                if should_count:
                    self.rep_count += 1
                
                # Update the verdict variable
                self.last_verdict = verdict

                # Reset the rep data
                self.reset_rep()

                # Return the calculated verdict
                return verdict
        else:
            self.live_feedback = "READY - START SQUAT"
    
    # A function that evaluated the users completed rep
    def evaluate_rep(self):
            # If they didn't hit depth, it's not a valid rep (No count)
            if not self.has_hit_bottom:
                return "TOO SHALLOW", False
            
            # If they have errors, it is a rep, but with a warning
            if self.errors_found:
                errors_str = ", ".join(self.errors_found)
                return f"REP OK (Fix: {errors_str})", True
                
            # If there are no errors, and they hit depth, return perfect rep
            return "PERFECT REP", True