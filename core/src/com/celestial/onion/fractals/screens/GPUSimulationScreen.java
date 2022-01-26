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
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

import static jcuda.driver.JCudaDriver.*;

public class GPUSimulationScreen implements Screen {
    private GDXFractals simulation;
    private Pixmap fractalMap;
    private Texture fractalTexture;
    Vector2 offset;
    float minX = -2.5f;
    float minY = -1.25f;
    float maxX = 1.5f;
    float maxY = 1.25f;
    float speed = 5f;
    int maxIt = 50;

    CUdeviceptr deviceOutput;
    CUfunction function;
    int cuNumElements = 800 * 800;

    GPUSimulationScreen(GDXFractals simulation) {
        this.simulation = simulation;
        fractalMap = new Pixmap(Gdx.graphics.getWidth(), Gdx.graphics.getHeight(), Pixmap.Format.RGBA8888);
        fractalMap.setFilter(Pixmap.Filter.NearestNeighbour);
        offset = new Vector2(0, 0);
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
        cuMemAlloc(deviceOutput, (long) cuNumElements * Sizeof.FLOAT);
        generateMandelbrotTexture();
    }

    public void generateMandelbrotTexture() {
        if(fractalTexture != null)
            fractalTexture.dispose();
        if(fractalMap != null)
            fractalMap.dispose();

        fractalMap = new Pixmap(Gdx.graphics.getWidth(), Gdx.graphics.getHeight(), Pixmap.Format.RGBA8888);

        Pointer kernelParams = Pointer.to(
                Pointer.to(new float[]{cuNumElements}),
                Pointer.to(new int[]{maxIt}),
                Pointer.to(new int[]{800}),
                Pointer.to(new int[]{800}),
                Pointer.to(new float[]{minX}),
                Pointer.to(new float[]{minY}),
                Pointer.to(new float[]{maxX}),
                Pointer.to(new float[]{maxY}),
                Pointer.to(new float[]{offset.x}),
                Pointer.to(new float[]{offset.y}),
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

        float[] hostOutput = new float[cuNumElements];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, (long) cuNumElements * Sizeof.FLOAT);

        for (int y = 0; y < Gdx.graphics.getHeight(); y++) {
            for (int x = 0; x < Gdx.graphics.getWidth(); x++) {
                int i = x + y * Gdx.graphics.getWidth();

                float n = hostOutput[i];
                fractalMap.drawPixel(x, y, Color.rgba8888(1, 1, 1, n / maxIt));
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
            speed *= 1.01f;
        }

        if (Gdx.input.isKeyPressed(Input.Keys.NUMPAD_SUBTRACT)) {
            minX *= 1.05f;
            maxX *= 1.05f;
            minY *= 1.05f;
            maxY *= 1.05f;
            speed *= 0.99f;
        }

        if (Gdx.input.isKeyPressed((Input.Keys.UP))) {
            offset.y -= speed;
        }

        if (Gdx.input.isKeyPressed((Input.Keys.RIGHT))) {
            offset.x += speed;
        }

        if (Gdx.input.isKeyPressed((Input.Keys.DOWN))) {
            offset.y += speed;
        }

        if (Gdx.input.isKeyPressed((Input.Keys.LEFT))) {
            offset.x -= speed;
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
