import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { ThemeToggle } from "../app/theme-toggle";

afterEach(() => { delete document.documentElement.dataset.theme; localStorage.clear(); });

describe("ThemeToggle", () => {
  it("switches to dark, remembers it, and switches back", async () => {
    const user = userEvent.setup();
    render(<ThemeToggle />);
    await user.click(screen.getByRole("button", { name: /toggle light or dark theme/i }));
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(localStorage.getItem("df-theme")).toBe("dark");
    expect(screen.getByText("Light mode")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /toggle light or dark theme/i }));
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(localStorage.getItem("df-theme")).toBe("light");
  });
  it("starts from a theme already set on the page", () => {
    document.documentElement.dataset.theme = "dark";
    render(<ThemeToggle />);
    expect(screen.getByText("Light mode")).toBeInTheDocument();
  });
});
