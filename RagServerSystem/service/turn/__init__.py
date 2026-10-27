"""轮次编排领域包：把原 turn_service.py 按职责拆成流式骨架、轮次状态、图编排、Agent 编排、RAG 编排五块。

对外仍通过 service/turn_service.py 这层兼容 facade 暴露，调用方无需改动。
"""
