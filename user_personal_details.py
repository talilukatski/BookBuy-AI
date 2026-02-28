from typing import List, Optional, Any, Dict


class UserPersonalDetails:
    """
    Holds personal preferences, reading history and purchase details.
    """
    def __init__(
        self,
        user_preferences: Optional[List[str]] = None,
        disliked_titles: Optional[List[str]] = None,
        already_read_titles: Optional[List[str]] = None,
        address: str = "Technion, Haifa",
        payment_token: str = "credit-card-1234"
    ):
        self.user_preferences = user_preferences or []
        self.disliked_titles = disliked_titles or []
        self.already_read_titles = already_read_titles or []

        # purchase details
        self.address = address
        self.payment_token = payment_token

    def initial_excluded_titles(self) -> List[str]:
        """Combine disliked and already read books into one exclusion list."""
        return list(set(self.disliked_titles + self.already_read_titles))
