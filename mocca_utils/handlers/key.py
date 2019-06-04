def key_press_handler(event):
    """ Hook up via `fig.canvas.events.key_press.connect(key_press_handler)` """
    if event.key.name == "Space":
        # Press Space to toggle pausing of program

        if not hasattr(event.source, "pause"):
            event.source.unfreeze()
            event.source.pause = True
            event.source.freeze()
        else:
            event.source.pause = not event.source.pause

        # Need this otherwise will throw run-time error
        event.source.events.key_press._emitting = False
        while event.source.pause:
            event.source.app.process_events()
