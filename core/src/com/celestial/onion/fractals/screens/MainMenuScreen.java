package com.celestial.onion.fractals.screens;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Screen;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.scenes.scene2d.*;
import com.badlogic.gdx.scenes.scene2d.ui.*;
import com.badlogic.gdx.scenes.scene2d.utils.ClickListener;
import com.badlogic.gdx.utils.Align;
import com.badlogic.gdx.utils.ScreenUtils;
import com.badlogic.gdx.utils.viewport.ScreenViewport;
import com.celestial.onion.fractals.GDXFractals;

import java.util.ArrayList;

public class MainMenuScreen implements Screen {
    private final GDXFractals simulation;
    private OrthographicCamera camera;
    private Stage stage;
    private Table table;
    private Label titleLabel;
    private Label authors;
    private ArrayList<TextButton> buttons;

    public MainMenuScreen(final GDXFractals simulation) {
        this.simulation = simulation;
        buttons = new ArrayList<>();
        camera = new OrthographicCamera();
        camera.setToOrtho(false, 800, 800);

        setupUi();
    }

    private void setupUi() {
        stage = new Stage(new ScreenViewport());
        Gdx.input.setInputProcessor(stage);

        table = new Table();
        table.setFillParent(true);
        table.align(Align.center);
        stage.addActor(table);

        table.row();
        titleLabel = new Label("Mandelbrot Set", new Label.LabelStyle(simulation.fontLabel, Color.WHITE));
        titleLabel.setAlignment(Align.top);
        table.add(titleLabel).pad(20);

        table.row();
        // start slimulation
        buttons.add(createButton("Simulate", new ClickListener() {
            @Override
            public void clicked(InputEvent event, float x, float y) {
                super.clicked(event, x, y);
                simulation.setScreen(new SimulationScreen(simulation));
                dispose();
            }
        }));

        buttons.add(createButton("Simulate (GPU)", new ClickListener() {
            @Override
            public void clicked(InputEvent event, float x, float y) {
                super.clicked(event, x, y);
                simulation.setScreen(new GPUSimulationScreen(simulation));
                dispose();
            }
        }));

        // close app
        buttons.add(createButton("Exit", new ClickListener() {
            @Override
            public void clicked(InputEvent event, float x, float y) {
                super.clicked(event, x, y);
                Gdx.app.exit();
            }
        }));

        for (TextButton b: buttons) {
            table.row();
            table.add(b).pad(10);
        }

        table.row();

        authors = new Label("2022 © Lukasz Mysliwiec & Kacper Grabiec", new Label.LabelStyle(
                simulation.fontDefault, Color.WHITE
        ));
        authors.setFontScale(0.7f);

        table.add(authors).pad(30);

        //table.setDebug(true);
    }

    /**
     *
     * @param label - set text to display
     * @param listener - set the listener
     * @return TextButton
     */
    private TextButton createButton(String label, EventListener listener) {
        TextButton.TextButtonStyle style = new TextButton.TextButtonStyle();
        style.font = simulation.fontDefault;

        TextButton button = new TextButton(label, style);
        button.addListener(listener);

        return button;
    }

    @Override
    public void show() {

    }

    @Override
    public void render(float delta) {
        ScreenUtils.clear(0.349f, 0.384f, 0.459f, 1);

        camera.update();
        simulation.batch.setProjectionMatrix(camera.combined);

//        simulation.batch.begin();
//        simulation.font.draw(simulation.batch, "Hello", 100, 400);
//        simulation.batch.end();

        stage.act(delta);
        stage.draw();
    }

    @Override
    public void resize(int width, int height) {
        stage.getViewport().update(width, height, true);
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
        stage.dispose();
    }
}
