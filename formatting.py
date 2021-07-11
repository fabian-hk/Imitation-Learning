tr = TrainingRun(
    ds=CommaAiDataSet(),
    input_size=(64, 60),
    output_bins=45,
    epochs=30,
    vis_ranges=[(0, 209), (2214, 2774), (4010, 4878)],
)
model, history = train(tr=tr)
