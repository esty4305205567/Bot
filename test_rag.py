"""בדיקה: מריץ שאלות על המאגר ומדווח. הרצה: python test_rag.py
השאלות כאן מותאמות לקבצי הדמה ב-data/. עם נתונים אמיתיים, החליפו אותן."""
import sys
import time

from rag import FamilyRAG

# (שאלה, האם התשובה אמורה להיות "אין מידע")
CASES = [
    ("כמה עלה האולם בבר המצווה של דוד?", False),
    ("מי היה הצלם בבר המצווה ומה המחיר שלו?", False),
    ("אילו מנהגים יש במשפחה בנוגע לבחירת סנדק בברית?", False),
    ("מה היה לוח הזמנים באירוסין של שרה ויעקב?", False),
    ("איזה ספק קייטרינג חזר ביותר משמחה אחת, ואיזה לקח הופק ממנו?", False),
    # שאלה שהתשובה אליה אינה במאגר:
    ("מה היה תפריט הסעודה בחתונה של הדוד בניו יורק?", True),
]


def main() -> int:
    rag = FamilyRAG()
    failures, report = 0, ["# דו\"ח בדיקות\n"]
    for i, (q, expect_no_info) in enumerate(CASES, start=1):
        t0 = time.time()
        ans = rag.ask(q)
        dt = time.time() - t0
        ok = ans.no_info if expect_no_info else (not ans.no_info and len(ans.sources) > 0)
        failures += 0 if ok else 1

        print(f"\n=== {i}. {q}  [{'PASS' if ok else 'FAIL'}] ({dt:.1f}s)")
        print(ans.text)
        for s in ans.sources:
            print(f"   - [{s.number}] {s.event_name} › {s.item_title} (score={s.score:.3f})")

        report.append(f"## {i}. {q}\n- תוצאה: {'PASS' if ok else 'FAIL'}\n- תשובה: {ans.text}")
        report += [f"- מקור: {s.event_name} › {s.item_title} (score={s.score:.3f})" for s in ans.sources]
        report.append("")

    with open("test_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n{len(CASES) - failures}/{len(CASES)} עברו. הדו\"ח נשמר ב-test_report.md")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
