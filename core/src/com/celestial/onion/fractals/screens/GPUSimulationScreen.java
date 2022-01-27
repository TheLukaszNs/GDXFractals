package com.celestial.onion.fractals.screens;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.Screen;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.Pixmap;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.utils.ScreenUtils;
import com.celestial.onion.fractals.GDXFractals;
import com.celestial.onion.fractals.Inspector;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import static jcuda.driver.JCudaDriver.*;

public class GPUSimulationScreen implements Screen {
    private GDXFractals simulation;
    private Pixmap fractalMap;
    private Texture fractalTexture;
    double offsetX = 0;
    double offsetY = 0;
    double minX = -2.5;
    double minY = -1.25;
    double maxX = 1.5;
    double maxY = 1.25;
    double speed = 5;
    int maxIt = 50;

    CUdeviceptr deviceOutput;
    CUfunction function;
    int cuNumElements = 800 * 800;

    GPUSimulationScreen(GDXFractals simulation) {
        this.simulation = simulation;
        fractalMap = new Pixmap(Gdx.graphics.getWidth(), Gdx.graphics.getHeight(), Pixmap.Format.RGBA8888);
        fractalMap.setFilter(Pixmap.Filter.NearestNeighbour);
    }

    @Override
    public void show() {
        setupCuda();
    }

    public void setupCuda() {
        JCudaDriver.setExceptionsEnabled(true);

        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, ".\\CUDA\\src\\VectorAddKern.ptx");
        function = new CUfunction();
        cuModuleGetFunction(function, module, "mandelbrot");

        deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, (long) cuNumElements * Sizeof.DOUBLE);
        generateMandelbrotTexture();
    }

    public void generateMandelbrotTexture() {
        if(fractalTexture != null)
            fractalTexture.dispose();
        if(fractalMap != null)
            fractalMap.dispose();

        fractalMap = new Pixmap(Gdx.graphics.getWidth(), Gdx.graphics.getHeight(), Pixmap.Format.RGBA8888);

        Pointer kernelParams = Pointer.to(
                Pointer.to(new int[]{cuNumElements}),
                Pointer.to(new int[]{maxIt}),
                Pointer.to(new int[]{800}),
                Pointer.to(new int[]{800}),
                Pointer.to(new double[]{minX}),
                Pointer.to(new double[]{minY}),
                Pointer.to(new double[]{maxX}),
                Pointer.to(new double[]{maxY}),
                Pointer.to(new double[]{offsetX}),
                Pointer.to(new double[]{offsetY}),
                Pointer.to(deviceOutput)
        );

        int blockSizeX = 512;
        int gridSizeX = (int)Math.ceil((double)cuNumElements / blockSizeX);

        cuLaunchKernel(function,
                gridSizeX, 1, 1,
                blockSizeX, 1, 1,
                0, null,
                kernelParams, null
        );

        double[] hostOutput = new double[cuNumElements];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, (long) cuNumElements * Sizeof.DOUBLE);

        for (int y = 0; y < Gdx.graphics.getHeight(); y++) {
            for (int x = 0; x < Gdx.graphics.getWidth(); x++) {
                int i = x + y * Gdx.graphics.getWidth();

                double n = hostOutput[i];
                fractalMap.drawPixel(x, y, Color.rgba8888(1, 1, 1, (float)n / maxIt));
            }
        }

        fractalTexture = new Texture(fractalMap);
    }

    @Override
    public void render(float delta) {
        generateMandelbrotTexture();

        if (Gdx.input.isKeyPressed(Input.Keys.NUMPAD_ADD)) {
            minX *= 0.95f;
            maxX *= 0.95f;
            minY *= 0.95f;
            maxY *= 0.95f;
            speed *= 0.95;
        }

        if (Gdx.input.isKeyPressed(Input.Keys.NUMPAD_SUBTRACT)) {
            minX *= 1.05f;
            maxX *= 1.05f;
            minY *= 1.05f;
            maxY *= 1.05f;
            speed *= 1.05f;
        }

        if (Gdx.input.isKeyPressed((Input.Keys.UP))) {
            offsetY -= speed * delta;
        }

        if (Gdx.input.isKeyPressed((Input.Keys.RIGHT))) {
            offsetX += speed * delta;
        }

        if (Gdx.input.isKeyPressed((Input.Keys.DOWN))) {
            offsetY += speed * delta;
        }

        if (Gdx.input.isKeyPressed((Input.Keys.LEFT))) {
            offsetX -= speed * delta;
        }

        if (Gdx.input.isKeyPressed((Input.Keys.Q))) {
            maxIt--;
        }

        if (Gdx.input.isKeyPressed((Input.Keys.E))) {
            maxIt++;
        }


        if (Gdx.input.isKeyPressed(Input.Keys.ESCAPE)) {
            simulation.setScreen(new MainMenuScreen(simulation));
            dispose();
        }

        ScreenUtils.clear(0.349f, 0.384f, 0.459f, 1.0f);

        simulation.batch.begin();
        simulation.batch.draw(fractalTexture, 0, 0);
        simulation.batch.end();
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
        cuMemFree(deviceOutput);
    }
}
