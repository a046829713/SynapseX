class InvalidModeError(Exception):
    """異常類，表示在不允許使用虛擬資料的模式中嘗試使用虛擬資料。"""

    def __init__(self, message="目前模式不可以使用虛擬數據，請重新確定"):
        # 初始化異常類，可以提供自訂訊息或使用預設訊息
        super().__init__(message)