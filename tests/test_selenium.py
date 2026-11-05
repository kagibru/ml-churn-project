from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def test_application():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get("http://localhost:8000/docs")

        assert "Churn Prediction API" in driver.title

    finally:
        driver.quit()
