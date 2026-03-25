import os
import sys
import shutil
import argparse
from typing import Optional

# ═══════════════════════════════════════════════════════════════
# 1. 路径自动配置 (保持与你原有逻辑一致)
# ═══════════════════════════════════════════════════════════════
CURRENT_DIR = os.getcwd()
# 尝试自动推断目录结构
if CURRENT_DIR.endswith("tutorial"):
    TUTORIAL_DIR = CURRENT_DIR
    GEAK_DIR = os.path.dirname(CURRENT_DIR)
else:
    # 假设当前目录包含 tutorial 或 geak_agent
    if os.path.exists(os.path.join(CURRENT_DIR, "tutorial")):
        TUTORIAL_DIR = os.path.join(CURRENT_DIR, "tutorial")
        GEAK_DIR = CURRENT_DIR
    elif os.path.exists(os.path.join(CURRENT_DIR, "geak_agent")):
        GEAK_DIR = CURRENT_DIR
        TUTORIAL_DIR = os.path.join(CURRENT_DIR, "tutorial")  # 假设存在
    else:
        #  fallback: 当前目录即为工作区
        TUTORIAL_DIR = CURRENT_DIR
        GEAK_DIR = os.path.dirname(CURRENT_DIR)

# 添加路径
for path in [GEAK_DIR, TUTORIAL_DIR]:
    if path not in sys.path and os.path.exists(path):
        sys.path.insert(0, path)


# ═══════════════════════════════════════════════════════════════
# 2. 核心运行函数
# ═══════════════════════════════════════════════════════════════
def run_geak_optimization(
    kernel_filename: str,
    output_dir_name: str = "outputs",
    iteration_num: int = 5,
    descendant_num: int = 3,
    temperature: float = 1.0,
    gpu_id: int = 0,
    target_mlu: str = "MLU590",
    api_key: Optional[str] = None,
    model_id: str = "gpt-4o",
    use_cambricon: bool = False,
    cambricon_api_key: Optional[str] = None,
    verbose: bool = True,
):
    """内部执行逻辑"""
    try:
        from tutorial_utils import (
            setup_environment,
            print_header,
            print_config,
            load_results,
            display_results_summary,
            display_generated_code,
        )
    except ImportError:
        print("❌ 错误: 无法导入 tutorial_utils。请确保脚本在正确的项目目录下运行。")
        print(f"   当前搜索路径: {sys.path}")
        sys.exit(1)

    # 初始化环境
    TUTORIAL_DIR, SRC_DIR, CORPUS_PATH, TutorialDataloader = setup_environment()

    if verbose:
        print_header("✓ GEAK-Agent Ready")

    # 配置 API
    final_api_key = api_key
    if use_cambricon:
        final_api_key = cambricon_api_key or os.getenv("CAMBRICON_API_KEY")
        final_model_id = "claude-sonnet-4"
        provider_name = "Cambricon"
    else:
        final_api_key = final_api_key or os.getenv("OPENAI_API_KEY")
        final_model_id = model_id
        provider_name = "OpenAI"

    # if not final_api_key:
    #    raise ValueError("❌ API Key 缺失。请使用 --api-key 参数或设置环境变量 OPENAI_API_KEY。")

    if verbose:
        print(f"✓ API: {provider_name} | Model: {final_model_id}")

    # 加载 Dataset
    print(f"🔍 正在加载 Kernel: {kernel_filename} ...")
    dataset = TutorialDataloader(
        kernel_names=[kernel_filename], corpus_path=CORPUS_PATH
    )

    if len(dataset) == 0:
        # 尝试给出更友好的错误提示
        possible_paths = [
            os.path.join(CORPUS_PATH, kernel_filename),
            os.path.join(TUTORIAL_DIR, kernel_filename),
            os.path.join(os.getcwd(), kernel_filename),
        ]
        print(f"❌ 错误: 在 Corpus 路径 ({CORPUS_PATH}) 下未找到 '{kernel_filename}'。")
        print("   请检查文件名是否正确，或文件是否已放入 corpus 目录。")
        sys.exit(1)

    # 初始化 Model & Agent
    from geak_agent.models.OpenAI import StandardOpenAIModel
    from geak_agent.agents.GaAgent import GaAgent

    model = StandardOpenAIModel(api_key=final_api_key, model_id=final_model_id)
    agent = GaAgent(
        model=model,
        dataset=dataset,
        corpus_path=CORPUS_PATH,
        descendant_num=descendant_num,
    )

    if verbose:
        print(f"✓ 已加载 {len(dataset)} 个 kernel | Agent 就绪")

    # 准备输出目录
    base_output_dir = os.path.join(TUTORIAL_DIR, output_dir_name)
    if os.path.exists(base_output_dir):
        if verbose:
            print(f"🧹 清理旧输出目录: {base_output_dir}")
        shutil.rmtree(base_output_dir)
    os.makedirs(base_output_dir, exist_ok=True)

    # 配置
    config = {
        "iteration_num": iteration_num,
        "temperature": temperature,
        "descendant_num": descendant_num,
        "ancestor_num": 5,
        "gpu_id": gpu_id,
        "target_mlu": target_mlu,
    }

    if verbose:
        print_config(config, f"🚀 开始优化: {kernel_filename}")

    # 运行
    result_path = os.path.join(base_output_dir, "tutorial_results.jsonl")

    try:
        agent.run(
            output_path=result_path,
            multi_thread=False,
            **config,
            start_iter=0,
            descendant_debug=1,
            profiling=True,
            start_idx=0,
        )
    except Exception as e:
        print(f"\n❌ 运行过程中出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n✅ 优化完成!")

    results, iteration = load_results(base_output_dir)
    display_results_summary(results, iteration)

    # View generated code
    display_generated_code(results)
    return base_output_dir


# ═══════════════════════════════════════════════════════════════
# 3. 命令行参数解析 (Argparse)
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="🚀 GEAK-Agent 通用 Kernel 优化工具",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 必需参数
    parser.add_argument(
        "kernel_file",
        type=str,
        help="待优化的 kernel 文件名 (例如: sin_kernel.py)，需位于 corpus 目录中",
    )

    # 可选参数 - 运行配置
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="outputs",
        help="输出结果目录名称 (默认: outputs)",
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=3, help="优化迭代次数 (默认: 5)"
    )
    parser.add_argument(
        "-d",
        "--descendants",
        type=int,
        default=3,
        help="每代生成的候选方案数 (默认: 3)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="LLM 温度参数 0.0-1.0 (默认: 1.0)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="使用的 GPU ID (默认: 0)")
    parser.add_argument(
        "--target", type=str, default="MLU590", help="目标硬件型号 (默认: MLU590)"
    )

    # 可选参数 - API 配置
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="模型 ID (默认: gpt-4o)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="XXXX",
        help="API Key (如果不传，将读取环境变量 OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--cambricon", action="store_true", help="启用寒武纪 API 模式 (默认使用 OpenAI)"
    )
    parser.add_argument(
        "--cambricon-key",
        type=str,
        default=None,
        help="寒武纪 API Key (如果不传，将读取环境变量 CAMBRICON_API_KEY)",
    )

    args = parser.parse_args()

    output_path = run_geak_optimization(
        kernel_filename=args.kernel_file,
        output_dir_name=args.output,
        iteration_num=args.iterations,
        descendant_num=args.descendants,
        temperature=args.temperature,
        gpu_id=args.gpu,
        target_mlu=args.target,
        api_key=args.api_key,
        model_id=args.model,
        use_cambricon=args.cambricon,
        cambricon_api_key=args.cambricon_key,
    )
    print(f"\n💾 结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
