import pandas as pd
import typer
from jonckheere_terpstra import jonckheere_terpstra_test
from rich.console import Console
from rich.table import Table

# 引入 statsmodels 用于 FDR 校正
from statsmodels.stats.multitest import multipletests

app = typer.Typer()
console = Console()


@app.command()
def jt_test(
    csv_file: str = typer.Argument(..., help="CSV 文件路径"),
    group_column: str = typer.Option(
        ..., "--group-col", help="分组列的名称 (例如 'Sample Type')"
    ),
    group_order: str = typer.Option(
        ..., "--order", help="分组顺序，用逗号分隔 (例如 'A,B,C,D')"
    ),
    id_column: str = typer.Option(
        "Sample Name", "--id-col", help="需要排除的样本ID列名"
    ),
    output_file: str = typer.Option(
        "jt_results_fdr.csv", "--output", help="结果保存路径"
    ),
    alternative: str = typer.Option(
        "two_sided", "--alt", help="假设检验方向 ('two_sided', 'greater', 'less')"
    ),
    continuity: bool = typer.Option(True, "--continuity", help="是否应用连续性校正"),
    fdr_alpha: float = typer.Option(0.05, "--fdr", help="FDR 显著性阈值 (默认 0.05)"),
):
    """
    批量执行 Jonckheere-Terpstra 趋势检验，并进行 FDR (Benjamini-Hochberg) 校正。
    """
    # 1. 读取数据
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        console.print(f"[bold red]读取 CSV 失败:[/bold red] {e}")
        raise typer.Exit(code=1)

    # 2. 验证列
    if group_column not in df.columns:
        console.print(f"[bold red]错误:[/bold red] 列 '{group_column}' 不存在。")
        raise typer.Exit(code=1)

    # 3. 处理分组
    ordered_groups = [g.strip() for g in group_order.split(",")]

    # 过滤数据并设置 Categorical 顺序
    df_filtered = df[df[group_column].isin(ordered_groups)].copy()
    df_filtered[group_column] = pd.Categorical(
        df_filtered[group_column], categories=ordered_groups, ordered=True
    )

    # 4. 筛选数值列（排除 ID 和 Group）
    cols_to_exclude = [group_column]
    if id_column in df.columns:
        cols_to_exclude.append(id_column)

    value_columns = df_filtered.select_dtypes(include=["number"]).columns.tolist()
    target_columns = [col for col in value_columns if col not in cols_to_exclude]

    if not target_columns:
        console.print("[bold yellow]警告:[/bold yellow] 没有找到数值列用于分析。")
        raise typer.Exit()

    console.print(
        f"正在分析 [bold green]{len(target_columns)}[/bold green] 个变量，分组顺序: {ordered_groups}"
    )

    # 5. 循环计算 Raw P-value
    results = []

    # 进度条表格预览（仅显示 Raw P）
    preview_table = Table(title="Test Progress (Preview)")
    preview_table.add_column("Variable", style="cyan")
    preview_table.add_column("Raw P-value", justify="right", style="magenta")

    with typer.progressbar(target_columns, label="Processing") as progress:
        for col in progress:
            try:
                # 提取非空数据
                current_data = df_filtered[[col, group_column]].dropna()

                if len(current_data) == 0:
                    continue

                x = current_data[col].values
                g = current_data[group_column].cat.codes.values

                jtrsum, pval, zstat = jonckheere_terpstra_test(
                    x=x, g=g, continuity=continuity, alternative=alternative
                )

                results.append(
                    {
                        "Variable": col,
                        "JTR_Sum": jtrsum,
                        "Z_statistic": zstat,
                        "P_value_Raw": pval,  # 原始 P 值
                    }
                )

                if len(results) <= 5:
                    preview_table.add_row(col, f"{pval:.4f}")

            except Exception:
                # 忽略计算错误的列（例如全为空值）
                continue

    # 6. 计算 FDR (Benjamini-Hochberg)
    results_df = pd.DataFrame(results)

    if not results_df.empty:
        # 使用 statsmodels 进行多重假设检验校正
        # method='fdr_bh' 即 Benjamini-Hochberg 方法
        reject, pvals_corrected, _, _ = multipletests(
            results_df["P_value_Raw"], alpha=fdr_alpha, method="fdr_bh"
        )

        results_df["FDR"] = pvals_corrected
        # 标记是否显著 (True/False)
        results_df[f"Sig_Raw (p<{fdr_alpha})"] = results_df["P_value_Raw"] < fdr_alpha
        results_df[f"Sig_FDR (q<{fdr_alpha})"] = results_df["FDR"] < fdr_alpha

        # 排序：先按是否 FDR 显著排序，再按 FDR 值从小到大排序
        results_df = results_df.sort_values(by=["FDR", "P_value_Raw"])

        # 7. 打印统计摘要
        n_raw_sig = results_df[f"Sig_Raw (p<{fdr_alpha})"].sum()
        n_fdr_sig = results_df[f"Sig_FDR (q<{fdr_alpha})"].sum()

        console.print("\n[bold]Summary:[/bold]")
        console.print(f"Total features tested: {len(results_df)}")
        console.print(
            f"Significant by Raw P-value (<{fdr_alpha}): [yellow]{n_raw_sig}[/yellow]"
        )
        console.print(
            f"Significant by FDR (<{fdr_alpha}):     [green]{n_fdr_sig}[/green]"
        )

        if n_raw_sig > 0:
            reduction = (1 - n_fdr_sig / n_raw_sig) * 100
            console.print(
                f"FDR correction reduced significant hits by [bold red]{reduction:.1f}%[/bold red]."
            )

        # 8. 保存结果
        # 将显著的列放在前面方便查看
        cols = [
            "Variable",
            "P_value_Raw",
            "FDR",
            f"Sig_FDR (q<{fdr_alpha})",
            "Z_statistic",
            "JTR_Sum",
        ]
        results_df[cols].to_csv(output_file, index=False)

        console.print(
            f"\n[bold green]Success![/bold green] Results with FDR saved to: [underline]{output_file}[/underline]"
        )
    else:
        console.print("[bold red]No results to process.[/bold red]")


@app.command()
def version() -> None:
    """Show version"""
    from sci_tools import __version__

    print(f"🔖 sci-tools {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
