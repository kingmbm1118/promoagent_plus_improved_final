"""
Monitor agent performance and prevent infinite loops
"""

import time
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentMonitor:
    """Monitor agent performance and detect stuck situations"""
    
    def __init__(self, timeout_seconds: int = 300, similarity_threshold: float = 0.9):
        self.start_time = None
        self.timeout_seconds = timeout_seconds
        self.similarity_threshold = similarity_threshold
        self.previous_outputs = []
        self.consecutive_similar_outputs = 0
        self.max_consecutive_similar = 3
        
    def start_monitoring(self):
        """Start the monitoring timer"""
        self.start_time = time.time()
        self.previous_outputs = []
        self.consecutive_similar_outputs = 0
        
    def check_timeout(self) -> bool:
        """Check if the operation has timed out"""
        if self.start_time is None:
            return False
            
        elapsed = time.time() - self.start_time
        return elapsed > self.timeout_seconds
        
    def record_output(self, output: str) -> Dict[str, Any]:
        """
        Record an output and check for repetition patterns
        
        Args:
            output: The agent output to record
            
        Returns:
            Dict with status information
        """
        if not self.previous_outputs:
            self.previous_outputs.append(output)
            return {"status": "ok", "message": "First output recorded"}
            
        # Check similarity with most recent output
        similarity = self._calculate_similarity(output, self.previous_outputs[-1])
        
        if similarity > self.similarity_threshold:
            self.consecutive_similar_outputs += 1
            message = f"Similar output detected ({self.consecutive_similar_outputs}/{self.max_consecutive_similar})"
            
            if self.consecutive_similar_outputs >= self.max_consecutive_similar:
                return {
                    "status": "stuck",
                    "message": "Agent appears to be stuck in a loop",
                    "similarity": similarity
                }
        else:
            self.consecutive_similar_outputs = 0
            message = "Output differs from previous"
            
        # Store output for future comparison
        self.previous_outputs.append(output)
        if len(self.previous_outputs) > 5:  # Keep only recent history
            self.previous_outputs.pop(0)
            
        return {
            "status": "ok", 
            "message": message,
            "similarity": similarity
        }
        
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        # Simple implementation - could be improved with more sophisticated algorithms
        if not str1 or not str2:
            return 0.0
            
        # Create sets of words
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_intervention_suggestion(self) -> str:
        """Get suggestion for intervention when agent is stuck"""
        suggestions = [
            "Simplify the model structure",
            "Focus on sequential flow first, then add choices and loops",
            "Try explicit intermediate variables for all operations",
            "Check ModelGenerator API usage",
            "Verify parameters for all method calls",
            "Fix indentation and syntax issues"
        ]
        
        return "\n".join([f"- {suggestion}" for suggestion in suggestions])