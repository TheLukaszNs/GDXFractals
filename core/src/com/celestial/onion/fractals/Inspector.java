package com.celestial.onion.fractals;

import imgui.ImGui;
import imgui.flag.ImGuiWindowFlags;

public class Inspector {

    private static final int _ITERATIONS_MAX_ = 1000;
    public int[] iterations;
    public float[] foregroundColor;
    public float[] borderColor;

    public Inspector(int[] iterations, float[] foregroundColor, float[] borderColor) {
        this.iterations = iterations;
        this.foregroundColor = foregroundColor;
        this.borderColor = borderColor;
    }

    public void show(Callable callable) {
        ImGui.begin("Select parameters", ImGuiWindowFlags.NoResize);
        ImGui.setWindowSize(400, 200);

        ImGui.colorEdit4("Border Color", borderColor);
        ImGui.colorEdit4("Foreground Color", foregroundColor);
        ImGui.sliderInt("Iterations", iterations, 1, _ITERATIONS_MAX_);
        ImGui.text("Use arrows to navigate the screen.");
        ImGui.text("Use + - to zoom in and zoom out.");
        ImGui.text("Press E to hide the inspector.");

        if (ImGui.button("Regenerate")) {
            callable.call();
        }

        ImGui.end();
    }

}
