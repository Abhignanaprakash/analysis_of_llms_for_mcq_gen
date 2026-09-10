from mcq_study.pipeline import normalize, render_target
def test_normalizes_labels_and_formats_target():
    row=normalize({"context":"<p> Context </p>","question":"Choose?","options":["one","two","three","four"],"answer":"b"})
    assert set(row["options"])=={"one","two","three","four"}
    assert row["options"]["ABCD".index(row["answer"])]=="two"
    assert f"Correct Answer: {row['answer']}" in render_target(row)
