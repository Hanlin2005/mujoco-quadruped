"""Procedural terrain generation for quadruped training.

Generates MuJoCo XML scene files with various terrain types:
- Flat ground (baseline)
- Random height field
- Stairs (up and down)
- Slopes
- Gaps
"""

import os
import numpy as np


class TerrainGenerator:
    """Generates MuJoCo XML files with procedural terrain."""

    TERRAIN_TYPES = ["flat", "rough", "stairs_up", "stairs_down", "slope", "gaps", "mixed"]

    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "models"
            )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, terrain_type="flat", seed=None, **kwargs):
        """Generate a terrain XML and return its path."""
        rng = np.random.RandomState(seed)

        if terrain_type == "flat":
            return self._generate_flat(**kwargs)
        elif terrain_type == "rough":
            return self._generate_rough(rng, **kwargs)
        elif terrain_type == "stairs_up":
            return self._generate_stairs(rng, direction="up", **kwargs)
        elif terrain_type == "stairs_down":
            return self._generate_stairs(rng, direction="down", **kwargs)
        elif terrain_type == "slope":
            return self._generate_slope(rng, **kwargs)
        elif terrain_type == "gaps":
            return self._generate_gaps(rng, **kwargs)
        elif terrain_type == "mixed":
            return self._generate_mixed(rng, **kwargs)
        else:
            raise ValueError(f"Unknown terrain type: {terrain_type}")

    def _xml_header(self):
        return """<mujoco model="go1 terrain scene">
  <include file="go1.xml"/>

  <statistic center="0 0 0.1" extent="0.8"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="obstacle" rgba="0.6 0.3 0.2 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
"""

    def _xml_footer(self):
        return """  </worldbody>
</mujoco>
"""

    def _generate_flat(self, **kwargs):
        """Flat ground - baseline terrain."""
        xml = self._xml_header() + self._xml_footer()
        path = os.path.join(self.output_dir, "terrain_flat.xml")
        with open(path, "w") as f:
            f.write(xml)
        return path

    def _generate_rough(self, rng, num_bumps=80, max_height=0.04, **kwargs):
        """Random small bumps on the ground."""
        xml = self._xml_header()
        for i in range(num_bumps):
            x = rng.uniform(0.5, 10.0)
            y = rng.uniform(-2.0, 2.0)
            h = rng.uniform(0.005, max_height)
            r = rng.uniform(0.05, 0.15)
            xml += f'    <geom type="cylinder" pos="{x:.3f} {y:.3f} {h/2:.4f}" size="{r:.3f} {h/2:.4f}" material="obstacle"/>\n'
        xml += self._xml_footer()
        path = os.path.join(self.output_dir, "terrain_rough.xml")
        with open(path, "w") as f:
            f.write(xml)
        return path

    def _generate_stairs(self, rng, direction="up", num_steps=8, step_height=0.04,
                         step_depth=0.3, step_width=1.5, **kwargs):
        """Staircase terrain."""
        xml = self._xml_header()
        for i in range(num_steps):
            x = 1.0 + i * step_depth
            if direction == "up":
                z = (i + 1) * step_height / 2
                h = (i + 1) * step_height
            else:
                z = (num_steps - i) * step_height / 2
                h = (num_steps - i) * step_height
            xml += (
                f'    <geom type="box" pos="{x:.3f} 0 {z:.4f}" '
                f'size="{step_depth/2:.3f} {step_width/2:.3f} {h/2:.4f}" material="obstacle"/>\n'
            )
        xml += self._xml_footer()
        name = f"terrain_stairs_{direction}.xml"
        path = os.path.join(self.output_dir, name)
        with open(path, "w") as f:
            f.write(xml)
        return path

    def _generate_slope(self, rng, angle_deg=10, length=3.0, width=1.5, **kwargs):
        """Sloped surface."""
        angle_rad = np.radians(angle_deg)
        xml = self._xml_header()
        # Create a tilted box
        mid_x = 2.0 + length / 2
        mid_z = np.sin(angle_rad) * length / 2
        xml += (
            f'    <geom type="box" pos="{mid_x:.3f} 0 {mid_z:.4f}" '
            f'euler="0 {-angle_deg} 0" '
            f'size="{length/2:.3f} {width/2:.3f} 0.02" material="obstacle"/>\n'
        )
        # Ramp approach
        xml += (
            f'    <geom type="box" pos="1.0 0 0.01" '
            f'size="0.5 {width/2:.3f} 0.01" material="obstacle"/>\n'
        )
        xml += self._xml_footer()
        path = os.path.join(self.output_dir, "terrain_slope.xml")
        with open(path, "w") as f:
            f.write(xml)
        return path

    def _generate_gaps(self, rng, num_gaps=5, gap_width=0.15, platform_width=0.6,
                       platform_height=0.03, **kwargs):
        """Platforms with gaps between them."""
        xml = self._xml_header()
        x = 1.0
        for i in range(num_gaps):
            xml += (
                f'    <geom type="box" pos="{x + platform_width/2:.3f} 0 {platform_height/2:.4f}" '
                f'size="{platform_width/2:.3f} 0.75 {platform_height/2:.4f}" material="obstacle"/>\n'
            )
            x += platform_width + gap_width
        xml += self._xml_footer()
        path = os.path.join(self.output_dir, "terrain_gaps.xml")
        with open(path, "w") as f:
            f.write(xml)
        return path

    def _generate_mixed(self, rng, **kwargs):
        """Mixed terrain with various obstacles."""
        xml = self._xml_header()
        x = 1.0

        # Section 1: Small bumps
        for i in range(20):
            bx = x + rng.uniform(0, 2.0)
            by = rng.uniform(-1.0, 1.0)
            h = rng.uniform(0.005, 0.03)
            r = rng.uniform(0.05, 0.12)
            xml += f'    <geom type="cylinder" pos="{bx:.3f} {by:.3f} {h/2:.4f}" size="{r:.3f} {h/2:.4f}" material="obstacle"/>\n'
        x += 2.5

        # Section 2: Small steps
        for i in range(4):
            step_h = (i + 1) * 0.025
            xml += (
                f'    <geom type="box" pos="{x:.3f} 0 {step_h/2:.4f}" '
                f'size="0.15 0.75 {step_h/2:.4f}" material="obstacle"/>\n'
            )
            x += 0.3

        x += 0.5

        # Section 3: Gaps
        for i in range(3):
            xml += (
                f'    <geom type="box" pos="{x:.3f} 0 0.015" '
                f'size="0.25 0.75 0.015" material="obstacle"/>\n'
            )
            x += 0.65

        xml += self._xml_footer()
        path = os.path.join(self.output_dir, "terrain_mixed.xml")
        with open(path, "w") as f:
            f.write(xml)
        return path

    def generate_all(self, seed=42):
        """Generate all terrain types and return dict of paths."""
        paths = {}
        for terrain_type in self.TERRAIN_TYPES:
            paths[terrain_type] = self.generate(terrain_type, seed=seed)
        return paths
