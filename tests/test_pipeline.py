from mcq_study.pipeline import normalize, render_target
def test_normalizes_labels_and_formats_target():
    row=normalize({"context":"<p> Context </p>","question":"Choose?","options":["one","two","three","four"],"answer":"b"})
    assert row["answer"]=="B" and row["options"]==["one","two","three","four"]
    assert "Correct Answer: B" in render_target(row)
