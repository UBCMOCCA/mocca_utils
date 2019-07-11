def key_press_handler(event):
    """ Hook up via `fig.canvas.events.key_press.connect(key_press_handler)` """
    if event.key.name == "Space":
        # Press Space to toggle pausing of program
        attr = "pause"

        if not hasattr(event.source, attr):
            event.source.unfreeze()
            setattr(event.source, attr, True)
            event.source.freeze()
        else:
            setattr(event.source, attr, not getattr(event.source, attr))

        # Need this otherwise will throw run-time error
        event.source.events.key_press._emitting = False
        while getattr(event.source, attr):
            event.source.app.process_events()

    elif event.key.name == "F1":
        # Save current frame to file

        import os
        import datetime
        from PIL import Image

        working_dir = os.getcwd()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = os.path.join(working_dir, timestamp + ".png")

        image = Image.fromarray(event.source.scene.canvas.render(), "RGBA")
        image.save(filename)
        print("Saved image at:", filename)

    elif event.key.name == "F2":
        # Toggle recording frame, save as video

        import os
        import datetime
        import moviepy.editor as mp

        callback_attr = "_recording_callback"
        buffer_attr = "_image_sequence"

        if not hasattr(event.source, callback_attr):
            # Start recording
            event.source.unfreeze()

            def _append_to_recording_buffer(event):
                buffer = getattr(event.source, buffer_attr)
                buffer.append(event.source.scene.canvas.render())

            setattr(event.source, callback_attr, _append_to_recording_buffer)
            setattr(event.source, buffer_attr, [])

            # Register callback to save frame into buffer
            event.source.events.draw.connect(_append_to_recording_buffer)
            event.source.freeze()
            print("[MoviePy] >>>> Start recording video... Press F2 again to stop.")

            # Uncomment if want video to look the same as real-time
            # Need to know FPS to know how to save video
            # Need to nop callback to make vispy not print fps
            # def _nop(x):
            #     pass

            # event.source.scene.canvas.measure_fps(callback=_nop)
        else:
            # Uncomment if want video to look the same as real-time
            # event.source.scene.canvas.measure_fps(callback=False)
            # fps = event.source.scene.canvas.fps

            # Stop recording
            working_dir = os.getcwd()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = os.path.join(working_dir, timestamp + ".png")
            event.source.freeze()

            buffer = getattr(event.source, buffer_attr)
            # Use real fps is required
            clip = mp.ImageSequenceClip(buffer, fps=60)

            working_dir = os.getcwd()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = os.path.join(working_dir, timestamp + ".mp4")
            clip.write_videofile(filename)

            # Clean up
            event.source.unfreeze()
            event.source.events.draw.disconnect(getattr(event.source, callback_attr))
            delattr(event.source, callback_attr)
            delattr(event.source, buffer_attr)
            event.source.freeze()
