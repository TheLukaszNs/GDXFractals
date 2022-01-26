package com.celestial.onion.fractals;

import com.badlogic.gdx.Game;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Graphics;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Window;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.g2d.freetype.FreeTypeFontGenerator;
import com.celestial.onion.fractals.screens.MainMenuScreen;
import imgui.ImGui;
import imgui.ImGuiIO;
import imgui.gl3.ImGuiImplGl3;
import imgui.glfw.ImGuiImplGlfw;
import imgui.lwjgl3.glfw.ImGuiImplGlfwNative;
import org.lwjgl.glfw.GLFWErrorCallback;

import static org.lwjgl.glfw.GLFW.glfwInit;

public class GDXFractals extends Game {
	public SpriteBatch batch;
	public BitmapFont fontLabel;
	public BitmapFont fontDefault;
	public ImGuiImplGlfw imGuiGlfw = new ImGuiImplGlfw();
	public ImGuiImplGl3 imGuiGl3 = new ImGuiImplGl3();
	public long windowHandle;

	@Override
	public void create () {
		GLFWErrorCallback.createPrint(System.err).set();
		if(!glfwInit()) {
			throw new IllegalStateException("Unable to initialize GLFW");
		}

		ImGui.createContext();
		final ImGuiIO io = ImGui.getIO();
		io.setIniFilename(null);

		windowHandle = ((Lwjgl3Graphics) Gdx.graphics).getWindow().getWindowHandle();
		imGuiGlfw.init(windowHandle, true);
		imGuiGl3.init("#version 130");

		batch = new SpriteBatch();
		setFont();

		this.setScreen(new MainMenuScreen(this));
	}

	public void setFont() {
		FreeTypeFontGenerator generator = new FreeTypeFontGenerator(Gdx.files.internal("font.ttf"));
		FreeTypeFontGenerator.FreeTypeFontParameter parameter = new FreeTypeFontGenerator.FreeTypeFontParameter();
		parameter.size = 36;
		parameter.color = Color.WHITE;
		fontLabel = generator.generateFont(parameter);
		parameter.size = 20;
		fontDefault = generator.generateFont(parameter);
		fontDefault.getRegion().getTexture().setFilter(Texture.TextureFilter.Linear, Texture.TextureFilter.Linear);
		generator.dispose();
	}

	@Override
	public void render () {
		super.render();
	}
	
	@Override
	public void dispose () {
		batch.dispose();
		fontLabel.dispose();
		imGuiGl3.dispose();
		imGuiGlfw.dispose();
		ImGui.destroyContext();
	}
}
