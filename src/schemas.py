from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """
    Input schema for customer churn prediction.
    """

    call_failure: int = Field(ge=0)
    complains: int = Field(ge=0, le=1)
    subscription_length: int = Field(ge=0)
    charge_amount: int = Field(ge=0, le=10)
    seconds_of_use: int = Field(ge=0)
    frequency_of_use: int = Field(ge=0)
    frequency_of_sms: int = Field(ge=0)
    distinct_called_numbers: int = Field(ge=0)

    age_group: int = Field(
        ge=1,
        le=5,
    )

    tariff_plan: int = Field(
        ge=1,
        le=2,
    )

    status: int = Field(
        ge=1,
        le=2,
    )

    age: int = Field(
        ge=0,
        le=120,
    )

    customer_value: float = Field(
        ge=0,
    )
