package com.celestial.onion.fractals.desktop;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Graphics;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;
import com.celestial.onion.fractals.GDXFractals;

public class DesktopLauncher {
	public static void main (String[] arg) {
		Lwjgl3ApplicationConfiguration config = new Lwjgl3ApplicationConfiguration();
		config.setTitle("Mandelbrot Set");
		config.setWindowedMode(800, 800);

		new Lwjgl3Application(new GDXFractals(), config);
	}
}
