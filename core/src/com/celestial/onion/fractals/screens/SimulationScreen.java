package com.celestial.onion.fractals.screens;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.Screen;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.Pixmap;

import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.utils.ScreenUtils;
import com.celestial.onion.fractals.Callable;
import com.celestial.onion.fractals.GDXFractals;
import com.celestial.onion.fractals.Inspector;
import com.celestial.onion.fractals.math.ComplexNumber;
import com.celestial.onion.fractals.math.Helpers;
import imgui.ImGui;

public class SimulationScreen implements Screen {
    private final int _ITERATIONS_MAX_ = 1000;

    private final GDXFractals simulation;
    private OrthographicCamera camera;

    private Pixmap fractalMap;
    private Texture fractalTexture;

    private float minX = -2.5f;
    private float maxX = 1.5f;
    private float minY = -1.25f;
    private float maxY = 1.25f;
    private Vector2 offset = new Vector2(0, 0);

    private boolean showUI = true;

    Inspector inspector;

    public SimulationScreen(GDXFractals simulation) {
        inspector = new Inspector(new int[_ITERATIONS_MAX_],
                new float[] {(float) 193/ 255, (float) 199 / 255,(float) 72 / 255, 1},
                new float[] {0, 0, 0, 1});

        this.simulation = simulation;

        fractalMap = new Pixmap(Gdx.graphics.getWidth(), Gdx.graphics.getHeight(), Pixmap.Format.RGBA8888);
        fractalMap.setFilter(Pixmap.Filter.NearestNeighbour);

        camera = new OrthographicCamera();
        camera.setToOrtho(false, Gdx.graphics.getWidth(), Gdx.graphics.getHeight());

        generate();
    }

    public void generate() {
        fractalMap = new Pixmap(Gdx.graphics.getWidth(), Gdx.graphics.getHeight(), Pixmap.Format.RGBA8888);

        for (int y = 0; y < Gdx.graphics.getHeight(); y++) {
            for (int x = 0; x < Gdx.graphics.getWidth(); x++) {
                ComplexNumber c = new ComplexNumber(0, 0);
                ComplexNumber p = new ComplexNumber(
                        Helpers.ScaleBetween(x + offset.x, 0, Gdx.graphics.getWidth(), minX, maxX),
                        Helpers.ScaleBetween(y + offset.y, 0, Gdx.graphics.getHeight(), minY, maxY)
                );

                int n = 0;
                while (n < inspector.iterations[0] && c.absoluteSquared() < 4) {
                    c = c.multiply(c).add(p);
                    n++;
                }

                if(c.absoluteSquared() >= 4) {
                    fractalMap.drawPixel(x, y, Color.rgba8888(
                            inspector.borderColor[0],
                            inspector.borderColor[1],
                            inspector.borderColor[2],
                            (float)n / inspector.iterations[0]
                    ));
                } else {
                    fractalMap.drawPixel(x, y, Color.rgba8888(
                            inspector.foregroundColor[0],
                            inspector.foregroundColor[1],
                            inspector.foregroundColor[2],
                            (float)n / inspector.iterations[0]
                    ));
                }

            }
        }

        fractalTexture = new Texture(fractalMap);
    }

    @Override
    public void show() {

    }

    @Override
    public void render(float delta) {
        if (Gdx.input.isKeyJustPressed(Input.Keys.E)) {
            showUI = !showUI;
        }

        if (Gdx.input.isKeyJustPressed(Input.Keys.F)) {
            offset = Vector2.Zero;
            minX = -2.5f;
            maxX = 1.5f;
            minY = -1.25f;
            maxY = 1.25f;
            generate();
        }

        if (Gdx.input.isKeyPressed(Input.Keys.ESCAPE)) {
            simulation.setScreen(new MainMenuScreen(simulation));
            dispose();
        }

        if (Gdx.input.isKeyPressed(Input.Keys.NUMPAD_ADD)) {
            minX *= 0.95f;
            maxX *= 0.95f;
            minY *= 0.95f;
            maxY *= 0.95f;
            generate();
        }

        if (Gdx.input.isKeyPressed(Input.Keys.NUMPAD_SUBTRACT)) {
            minX *= 1.05f;
            maxX *= 1.05f;
            minY *= 1.05f;
            maxY *= 1.05f;
            generate();
        }

        if (Gdx.input.isKeyPressed((Input.Keys.UP))) {
            offset.y -= 5;
            generate();
        }

        if (Gdx.input.isKeyPressed((Input.Keys.RIGHT))) {
            offset.x += 5;
            generate();
        }

        if (Gdx.input.isKeyPressed((Input.Keys.DOWN))) {
            offset.y += 5;
            generate();
        }

        if (Gdx.input.isKeyPressed((Input.Keys.LEFT))) {
            offset.x -= 5;
            generate();
        }

        ScreenUtils.clear(0.349f, 0.384f, 0.459f, 1.0f);

        camera.update();
        simulation.batch.setProjectionMatrix(this.camera.combined);

        simulation.batch.begin();
        simulation.batch.draw(fractalTexture, 0, 0);
        simulation.batch.end();

        simulation.imGuiGlfw.newFrame();
        ImGui.newFrame();

        if (showUI) {
            inspector.show(this::generate);
        }

        ImGui.render();
        simulation.imGuiGl3.renderDrawData(ImGui.getDrawData());
    }

    @Override
    public void resize(int width, int height) {

    }

    @Override
    public void pause() {

    }

    @Override
    public void resume() {

    }

    @Override
    public void hide() {

    }

    @Override
    public void dispose() {
        fractalMap.dispose();
        fractalTexture.dispose();
    }
}