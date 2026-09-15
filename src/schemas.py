from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    call_failure: int = Field(
        ge=0,
        description="Number of call failures",
    )

    complains: int = Field(
        ge=0,
        le=1,
        description="Complaint indicator: 0 or 1",
    )

    subscription_length: int = Field(
        ge=0,
        description="Customer subscription length",
    )

    charge_amount: int = Field(
        ge=0,
        le=10,
        description="Encoded charge amount",
    )

    seconds_of_use: int = Field(
        ge=0,
        description="Total seconds of service usage",
    )

    frequency_of_use: int = Field(
        ge=0,
        description="Frequency of service usage",
    )

    frequency_of_sms: int = Field(
        ge=0,
        description="Frequency of SMS usage",
    )

    distinct_called_numbers: int = Field(
        ge=0,
        description="Number of distinct called numbers",
    )

    age_group: int = Field(
        ge=1,
        le=5,
        description="Encoded age group from 1 to 5",
    )

    tariff_plan: int = Field(
        ge=1,
        le=2,
        description="Encoded tariff plan: 1 or 2",
    )

    status: int = Field(
        ge=1,
        le=2,
        description="Encoded customer status: 1 or 2",
    )

    age: int = Field(
        ge=0,
        le=120,
        description="Customer age",
    )

    customer_value: float = Field(
        ge=0,
        description="Calculated customer value",
    )