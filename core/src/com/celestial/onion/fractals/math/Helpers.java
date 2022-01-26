package com.celestial.onion.fractals.math;

public class Helpers {
    public static double ScaleBetween(double num, double min, double max, double newMin, double newMax) {
        return ((newMax - newMin)*(num - min))/(max - min) + newMin;
    }
}
