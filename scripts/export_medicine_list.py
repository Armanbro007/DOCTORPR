from __future__ import annotations

from src.dataset import load_all_labels


def main() -> None:
    df = load_all_labels()
    names = sorted(df["medicine_name"].dropna().astype(str).str.strip().unique())
    with open("medicine_list.csv", "w", encoding="utf-8") as handle:
        handle.write("medicine_name\n")
        for name in names:
            if name:
                handle.write(f"{name}\n")
    print(f"Saved {len(names)} medicine names to medicine_list.csv")


if __name__ == "__main__":
    main()
