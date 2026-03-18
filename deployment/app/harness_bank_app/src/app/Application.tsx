import * as React from "react";
import {
  Box,
  Button,
  CssBaseline,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControl,
  FormControlLabel,
  FormLabel,
  TextField,
  Typography,
  Stack,
  Checkbox,
  Select,
  MenuItem,
  Alert,
  InputAdornment,
} from "@mui/material";
import { SelectChangeEvent } from "@mui/material/Select";
import MuiCard from "@mui/material/Card";
import { styled } from "@mui/material/styles";

import AppTheme from "../shared-theme/AppTheme";
import { HarnessIcon } from "./CustomIcons";
import ColorModeSelect from "../shared-theme/ColorModeSelect";
import environment from '../environments/environment.index';

const Card = styled(MuiCard)(({ theme }) => ({
  display: "flex",
  flexDirection: "column",
  alignSelf: "center",
  width: "100%",
  padding: theme.spacing(4),
  gap: theme.spacing(2),
  margin: "auto",
  boxShadow:
    "hsla(220, 30%, 5%, 0.05) 0px 5px 15px 0px, hsla(220, 25%, 10%, 0.05) 0px 15px 35px -5px",
  [theme.breakpoints.up("sm")]: {
    width: "450px",
  },
  ...theme.applyStyles("dark", {
    boxShadow:
      "hsla(220, 30%, 5%, 0.5) 0px 5px 15px 0px, hsla(220, 25%, 10%, 0.08) 0px 15px 35px -5px",
  }),
}));

const ApplicationContainer = styled(Stack)(({ theme }) => ({
  height: "calc((1 - var(--template-frame-height, 0)) * 100dvh)",
  minHeight: "100%",
  padding: theme.spacing(2),
  [theme.breakpoints.up("sm")]: {
    padding: theme.spacing(4),
  },
  "&::before": {
    content: "''",
    display: "block",
    position: "absolute",
    zIndex: -1,
    inset: 0,
    backgroundImage:
      "radial-gradient(ellipse at 50% 50%, hsl(210, 100%, 97%), hsl(0, 0%, 100%))",
    backgroundRepeat: "no-repeat",
    ...theme.applyStyles("dark", {
      backgroundImage:
        "radial-gradient(at 50% 50%, hsla(210, 100%, 16%, 0.5), hsl(220, 30%, 5%))",
    }),
  },
}));

export default function Application(props: { disableCustomTheme?: boolean }) {
  const [numChildren, setNumChildren] = React.useState("0");
  const [responseMessage, setResponseMessage] = React.useState("");
  const [dialogOpen, setDialogOpen] = React.useState(false);
  const [dialogSeverity, setDialogSeverity] = React.useState<"success" | "warning" | "error">("success");
  const [emailError, setEmailError] = React.useState(false);
  const [emailErrorMessage, setEmailErrorMessage] = React.useState("");
  const [incomeError, setIncomeError] = React.useState(false);
  const [incomeErrorMessage, setIncomeErrorMessage] = React.useState("");
  const [nameError, setNameError] = React.useState(false);
  const [nameErrorMessage, setNameErrorMessage] = React.useState("");

  const validateInputs = () => {
    const name = document.getElementById("name") as HTMLInputElement;
    const email = document.getElementById("email") as HTMLInputElement;
    const income = document.getElementById("income") as HTMLInputElement;

    let isValid = true;

    if (!name.value || name.value.length < 1) {
      setNameError(true);
      setNameErrorMessage("Name is required.");
      isValid = false;
    } else {
      setNameError(false);
      setNameErrorMessage("");
    }

    if (!email.value || !/\S+@\S+\.\S+/.test(email.value)) {
      setEmailError(true);
      setEmailErrorMessage("Please enter a valid email address.");
      isValid = false;
    } else {
      setEmailError(false);
      setEmailErrorMessage("");
    }

    if (!income.value || Number(income.value)< 60000) {
      setIncomeError(true);
      setIncomeErrorMessage("Must have a minimum income of $60,000");
      isValid = false;
    } else {
      setIncomeError(false);
      setIncomeErrorMessage("");
    }

    return isValid;
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
  
    const income = document.getElementById("income") as HTMLInputElement;
    const ownCar = document.getElementById("ownCar") as HTMLInputElement;
    const ownHouse = document.getElementById("ownHouse") as HTMLInputElement;

    console.log({
      income: income?.value,
      children: numChildren,
      ownCar: ownCar?.checked,
      ownHouse: ownHouse?.checked,
    });
  
    const requestData = {
      income: income.value,
      children: numChildren,
      ownCar: ownCar?.checked,
      ownHouse: ownHouse?.checked,
    };
  
    try {
      const response = await fetch(`${environment.apiUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData),
      });
  
      if (response.ok) {
        const result = await response.json();
        if (result.prediction === 1) {
          setDialogSeverity("success");
          setResponseMessage("Congratulations, your application has been approved!");
        } else if (result.prediction === 0) {
          setDialogSeverity("warning");
          setResponseMessage("Unfortunately, your application was not approved this time.");
        }
      } else {
        setResponseMessage("Failed to get a prediction. Please try again.");
        setDialogSeverity("error");
      }
    } catch (error) {
      if (error instanceof Error) {
        setResponseMessage(`Error: ${error.message}`);
      } else {
        setResponseMessage(`An unknown error occurred.`);
      }
      setDialogSeverity("error");
    } finally {
      setDialogOpen(true);
    }
  };

  const handleChange = (event: SelectChangeEvent) => {
    setNumChildren(event.target.value as string);
  };

  const handleClose = () => {
    setDialogOpen(false);
  };

  return (
    <AppTheme {...props}>
      <CssBaseline enableColorScheme />
      <ColorModeSelect sx={{ position: "fixed", top: "1rem", right: "1rem" }} />
      <ApplicationContainer direction="column" justifyContent="space-between">
        <Card variant="outlined">
          <HarnessIcon />
          <Typography
            component="h1"
            variant="h4"
            sx={{ width: "100%", fontSize: "clamp(2rem, 10vw, 2.15rem)" }}
          >
            Credit Card Application
          </Typography>
          <Box
            component="form"
            onSubmit={handleSubmit}
            sx={{ display: "flex", flexDirection: "column", gap: 2 }}
          >
            <FormControl fullWidth>
              <FormLabel htmlFor="name">Full name</FormLabel>
              <TextField
                required
                name="name"
                id="name"
                autoComplete="name"
                placeholder="Captain Canary"
                error={nameError}
                helperText={nameErrorMessage}
                color={nameError ? "error" : "primary"}
              />
            </FormControl>
            <FormControl fullWidth>
              <FormLabel htmlFor="email">Email Address</FormLabel>
              <TextField
                required
                name="email"
                id="email"
                autoComplete="email"
                placeholder="captain_canary@harness.io"
                error={emailError}
                helperText={emailErrorMessage}
                color={incomeError ? "error" : "primary"}
              />
            </FormControl>
            <FormControl fullWidth>
              <FormLabel htmlFor="income">Annual Income</FormLabel>
              <TextField
                required
                name="income"
                id="income"
                placeholder="0"
                error={incomeError}
                helperText={incomeErrorMessage}
                color={incomeError ? "error" : "primary"}
                slotProps={{
                  input: {
                    startAdornment: <InputAdornment position="start">$</InputAdornment>,
                  },
                }}
              />
            </FormControl>
            <Box sx={{ minWidth: 120 }}>
              <FormControl fullWidth>
                <FormLabel htmlFor="children">Number of Children</FormLabel>
                <Select
                  name="children"
                  id="children"
                  value={numChildren}
                  onChange={handleChange}
                >
                  <MenuItem value={0}><em>None</em></MenuItem>
                  <MenuItem value={1}>1</MenuItem>
                  <MenuItem value={2}>2</MenuItem>
                  <MenuItem value={3}>3</MenuItem>
                  <MenuItem value={4}>4</MenuItem>
                  <MenuItem value={5}>5</MenuItem>
                  <MenuItem value={6}>6</MenuItem>
                </Select>
              </FormControl>
            </Box>
            <FormControlLabel
              control={
                <Checkbox
                  id="ownCar"
                  value="ownCar"
                  color="primary"
                />
              }
              label="Own Car"
            />
            <FormControlLabel
              control={
                <Checkbox
                  id="ownHouse"
                  value="ownHouse"
                  color="primary"
                />
              }
              label="Own House"
            />
            <Divider></Divider>
            <FormControlLabel
              required
              control={
                <Checkbox
                  value="agreeTerms"
                  color="primary"
                />}
              label="I have read and agree to the terms and conditions."
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              onClick={validateInputs}
            >
              Submit Application
            </Button>
          </Box>
        </Card>
      </ApplicationContainer>
      <Dialog open={dialogOpen} onClose={handleClose}>
        <DialogTitle>Application Result</DialogTitle>
        <DialogContent sx={{ padding: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box sx={{ marginY: 2 }}>
            <Alert
              variant="outlined"
              severity={dialogSeverity}
            >
              {responseMessage}
            </Alert>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose} autoFocus>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </AppTheme>
  );
}
