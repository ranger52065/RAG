import argparse

from agent.rag_agent import agent_with_history


def run_interactive():
    print("RAG知识库问答系统")
    print("输入 'q', 'exit', 'quit' 退出")
    print("-" * 40)

    session_id = "default_session"

    while True:
        question = input("用户：")
        if question.lower() in ['q', 'exit', 'quit']:
            print("对话结束，拜拜！")
            break

        response = agent_with_history.invoke(
            {'input': question},
            config={'configurable': {"session_id": session_id}}
        )
        print(f"助手：{response.get('output', response)}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAG企业知识库系统')
    parser.add_argument('--mode', choices=['interactive', 'graph'], default='interactive',
                        help='运行模式: interactive (交互式问答) 或 graph (LangGraph工作流)')
    args = parser.parse_args()

    if args.mode == 'graph':
        from graph2.graph_2 import graph
        print("启动LangGraph工作流模式...")
        question = input('用户：')
        last_value = None
        for output in graph.stream({"question": question}):
            for key, value in output.items():
                last_value = value
                print(f"Node '{key}':")
            print("\n---\n")
        if last_value and 'generation' in last_value:
            print(last_value["generation"])
    else:
        run_interactive()
