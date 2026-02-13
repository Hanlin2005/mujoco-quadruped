#!/usr/bin/env python3
"""Quick setup and verification script."""

import os
import sys


def check_imports():
    """Verify all required packages are importable."""
    packages = [
        ("mujoco", "MuJoCo"),
        ("gymnasium", "Gymnasium"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("tensorboard", "TensorBoard"),
        ("imageio", "ImageIO"),
        ("yaml", "PyYAML"),
    ]

    all_ok = True
    for module_name, display_name in packages:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {display_name}: {version}")
        except ImportError:
            print(f"  [FAIL] {display_name}: not installed")
            all_ok = False
    return all_ok


def check_model():
    """Verify the Go1 model loads correctly."""
    import mujoco

    model_path = os.path.join(os.path.dirname(__file__), "models", "scene.xml")
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        print(f"  [OK] Go1 model loaded ({model.nu} actuators, {model.nq} qpos)")
        return True
    except Exception as e:
        print(f"  [FAIL] Model load error: {e}")
        return False


def check_env():
    """Verify the custom environment works."""
    sys.path.insert(0, os.path.dirname(__file__))
    from envs.quadruped_env import QuadrupedEnv

    try:
        env = QuadrupedEnv()
        obs, info = env.reset(seed=42)
        print(f"  [OK] Environment created (obs shape: {obs.shape})")

        # Test a few steps
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

        print(f"  [OK] Environment stepping works (reward: {reward:.3f})")
        env.close()
        return True
    except Exception as e:
        print(f"  [FAIL] Environment error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 50)
    print("Quadruped Locomotion Project - Setup Check")
    print("=" * 50)

    print("\n1. Checking dependencies...")
    deps_ok = check_imports()

    print("\n2. Checking robot model...")
    model_ok = check_model()

    print("\n3. Checking custom environment...")
    env_ok = check_env()

    print("\n" + "=" * 50)
    if deps_ok and model_ok and env_ok:
        print("All checks passed! Ready to train.")
        print("\nQuick start:")
        print("  python training/train.py --config training/config.yaml")
    else:
        print("Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
