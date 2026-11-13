from src.processing import DataProcessor


def _print_top_segments(stats, limit=3):
    for name, row in stats.head(limit).iterrows():
        print(
            f"  - {name}: {int(row['users']):,} чел. "
            f"({row['share']:.1%}), возврат {row['return_rate']:.1%}"
        )


def main():
    processor = DataProcessor()
    processor.load_data()
    processor.load_currency_rates()
    processor.preprocess()
    processor.build_profiles()
    processor.filter_active_profiles()

    segments = processor.retention_by_segments()
    hypotheses = processor.check_hypotheses(segments)
    revenue = processor.analyze_revenue()
    weekday = processor.analyze_weekday(segments)
    correlation = processor.correlation_analysis()
    profile_segments = processor.profile_segments()
    preprocessing = processor.preprocessing_stats
    profile_filter = processor.profile_filter_stats
    profiles = processor.profiles
    final_mean_revenue = processor.data["revenue_rub"].mean()
    final_median_revenue = processor.data["revenue_rub"].median()

    print("=" * 72)
    print("АНАЛИЗ ЛОЯЛЬНОСТИ ПОЛЬЗОВАТЕЛЕЙ ЯНДЕКС АФИШИ")
    print("=" * 72)

    print("\nДанные и подготовка")
    print(
        f"  - Исходно: {preprocessing['initial_rows']:,} заказов, "
        f"{preprocessing['initial_users']:,} пользователей."
    )
    print(
        f"  - Удалено неявных дубликатов: {preprocessing['duplicates']:,}; "
        f"тенге пересчитано: {preprocessing['kzt_orders']:,} заказов."
    )
    print(
        f"  - Удалено отрицательной выручки: {preprocessing['negative_count']:,}; "
        f"выше 99-го перцентиля ({preprocessing['revenue_p99']:,.2f} руб.): "
        f"{preprocessing['above_p99_count']:,}."
    )
    print(
        f"  - После очистки: {preprocessing['final_rows']:,} заказов, "
        f"{preprocessing['final_users']:,} пользователей; период "
        f"{preprocessing['period_start']:%d.%m.%Y} — "
        f"{preprocessing['period_end']:%d.%m.%Y}."
    )
    print(
        f"  - Удалено аномально активных пользователей: "
        f"{profile_filter['removed_users']:,} "
        f"(выше {profile_filter['orders_p99']:.0f} заказов)."
    )
    print(
        f"  - Итоговая выборка: {len(profiles):,} пользователей, "
        f"{len(processor.data):,} заказов."
    )

    print("\nПрофиль пользователей")
    print(
        f"  - Средняя выручка с заказа: {final_mean_revenue:,.2f} руб.; "
        f"медиана: {final_median_revenue:,.2f} руб."
    )
    print(
        f"  - Доля пользователей с 2+ заказами: {profiles['is_two'].mean():.2%}; "
        f"среднее число заказов: {profiles['total_orders'].mean():.2f}; "
        f"медиана: {profiles['total_orders'].median():.0f}."
    )
    print(
        f"  - Средний интервал между заказами у вернувшихся: "
        f"{profiles['avg_days_between_orders'].mean():.1f} дн.; "
        f"медиана: {profiles['avg_days_between_orders'].median():.1f} дн."
    )
    print("  - Сегменты по числу заказов:")
    for segment, row in profile_segments.iterrows():
        print(
            f"    {segment}: {int(row['users']):,} чел. "
            f"({row['user_share']:.1%} пользователей, "
            f"{row['order_share']:.1%} заказов)"
        )

    print("\nТочки входа и возврат")
    print("  - Тип первого мероприятия:")
    _print_top_segments(segments["first_event_type"])
    print("  - Устройство первого заказа:")
    _print_top_segments(segments["first_device"])
    print("  - Регионы первого заказа:")
    _print_top_segments(segments["first_region"])
    print(
        f"  - Средний возврат по выборке: {profiles['is_two'].mean():.2%}; "
        f"доля топ-10 регионов: "
        f"{segments['first_region'].head(10)['share'].sum():.1%}."
    )

    print("\nПроверка гипотез")
    print(
        f"  - Спорт: {hypotheses['sport_rate']:.2%}, концерты: "
        f"{hypotheses['concerts_rate']:.2%}; разница "
        f"{hypotheses['sport_diff'] * 100:+.2f} п.п. "
        "Гипотеза не подтвердилась."
    )
    print(
        f"  - Топ-10 регионов: {hypotheses['top_rate']:.2%}, остальные: "
        f"{hypotheses['rest_rate']:.2%}; разница "
        f"{hypotheses['region_diff'] * 100:+.2f} п.п. "
        "Гипотеза подтвердилась."
    )

    print("\nВыручка и день недели")
    print(
        f"  - Средняя выручка: 1 заказ — {revenue['single_mean']:,.2f} руб., "
        f"2+ — {revenue['repeated_mean']:,.2f} руб."
    )
    print(
        f"  - Медиана: 1 заказ — {revenue['single_median']:,.2f} руб., "
        f"2+ — {revenue['repeated_median']:,.2f} руб."
    )
    weekday_stats = weekday["stats"]
    print(
        f"  - Возврат по дням недели: от {weekday_stats['return_rate'].min():.2%} "
        f"до {weekday_stats['return_rate'].max():.2%}; "
        f"размах {weekday['spread'] * 100:.2f} п.п."
    )

    print("\nКорреляционный анализ (phi_k)")
    for name, value in correlation["total_orders"].head(5).items():
        print(f"  - {name}: {value:.3f}")
    for segment in ["3–4 заказа", "5+ заказов"]:
        top = correlation["segment_results"][segment].sort_values(ascending=False)
        print(f"  - Сегмент {segment}:")
        for name, value in top.head(3).items():
            print(f"    {name}: {value:.3f}")

    print("\nВыводы и рекомендации")
    print(
        "  - Ядро выручки — пользователи с 5+ заказами: они создают "
        f"{profile_segments.loc['5+', 'order_share']:.1%} заказов."
    )
    print(
        "  - Приоритет удержания — крупные регионы, театр и выставки; "
        "для спорта нужны отдельные механики повторного вовлечения."
    )
    print(
        "  - Вторую покупку важно стимулировать в первые две недели: "
        "медианный интервал у вернувшихся составляет "
        f"{profiles['avg_days_between_orders'].median():.1f} дн."
    )
    print(
        "  - Desktop-аудитория возвращается чаще mobile; стоит перенести "
        "лучшие практики desktop-пути в мобильный сценарий."
    )
    print(
        "  - День недели и тип первого мероприятия слабо связаны с возвратом; "
        "для прогноза использовать признаки, доступные на момент первого заказа."
    )


if __name__ == "__main__":
    main()
